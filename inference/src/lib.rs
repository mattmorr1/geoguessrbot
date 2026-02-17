use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

const MILVUS_URL: &str = "http://localhost:19530";
const DB_NAME: &str = "geoguessr";
const COLLECTION: &str = "world_locations";
const TOP_K: usize = 5;
const TOP_N: usize = 3;

#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    #[pyo3(get)]
    pub lat: f64,
    #[pyo3(get)]
    pub lon: f64,
    #[pyo3(get)]
    pub confidence_pct: f64,
    #[pyo3(get)]
    pub s2_token: String,
}

#[pymethods]
impl Prediction {
    fn __repr__(&self) -> String {
        format!(
            "Prediction(lat={:.4}, lon={:.4}, confidence={:.2}%, s2={})",
            self.lat, self.lon, self.confidence_pct, self.s2_token
        )
    }
}

#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    #[pyo3(get)]
    pub predictions: Vec<Prediction>,
    #[pyo3(get)]
    pub weighted_lat: f64,
    #[pyo3(get)]
    pub weighted_lon: f64,
}

#[pymethods]
impl InferenceResult {
    fn __repr__(&self) -> String {
        format!(
            "InferenceResult(weighted=({:.4}, {:.4}), predictions={})",
            self.weighted_lat,
            self.weighted_lon,
            self.predictions.len()
        )
    }

    fn to_dict(&self) -> PyResult<String> {
        serde_json::to_string_pretty(self).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Serialization error: {}", e))
        })
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SearchRequest {
    db_name: String,
    collection_name: String,
    data: Vec<Vec<f32>>,
    anns_field: String,
    limit: usize,
    output_fields: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct MilvusResponse {
    code: i32,
    data: Option<Vec<MilvusHit>>,
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MilvusHit {
    distance: f64,
    gps: GpsField,
    s2sphere_boundary: String,
}

#[derive(Debug, Deserialize)]
struct GpsField {
    lat: f64,
    lon: f64,
}

pub async fn search_milvus(query_vec: Vec<f32>) -> Result<Vec<MilvusHit>, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();

    let body = SearchRequest {
        db_name: DB_NAME.to_string(),
        collection_name: COLLECTION.to_string(),
        data: vec![query_vec],
        anns_field: "alphaearth_vec".to_string(),
        limit: TOP_K,
        output_fields: vec!["gps".to_string(), "s2sphere_boundary".to_string()],
    };

    let resp = client
        .post(format!("{}/v2/vectordb/entities/search", MILVUS_URL))
        .header("Authorization", "Bearer root:Milvus")
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await?;

    let milvus_resp: MilvusResponse = resp.json().await?;

    if milvus_resp.code != 0 {
        return Err(format!(
            "Milvus error (code {}): {}",
            milvus_resp.code,
            milvus_resp.message.unwrap_or_default()
        )
        .into());
    }

    milvus_resp
        .data
        .ok_or_else(|| "Milvus returned no data".into())
}

pub fn weighted_interpolation(hits: &[MilvusHit]) -> InferenceResult {
    let weights: Vec<f64> = hits.iter().map(|h| h.distance.max(0.0)).collect();
    let total_weight: f64 = weights.iter().sum();

    let (weighted_lat, weighted_lon) = if total_weight > 0.0 {
        let lat = weights
            .iter()
            .zip(hits.iter())
            .map(|(w, h)| w * h.gps.lat)
            .sum::<f64>()
            / total_weight;
        let lon = weights
            .iter()
            .zip(hits.iter())
            .map(|(w, h)| w * h.gps.lon)
            .sum::<f64>()
            / total_weight;
        (lat, lon)
    } else {
        (hits[0].gps.lat, hits[0].gps.lon)
    };

    let top_n = &hits[..TOP_N.min(hits.len())];
    let top_n_weights: Vec<f64> = top_n.iter().map(|h| h.distance.max(0.0)).collect();
    let top_n_sum: f64 = top_n_weights.iter().sum();

    let predictions: Vec<Prediction> = top_n
        .iter()
        .zip(top_n_weights.iter())
        .map(|(hit, &w)| Prediction {
            lat: hit.gps.lat,
            lon: hit.gps.lon,
            confidence_pct: if top_n_sum > 0.0 {
                (w / top_n_sum) * 100.0
            } else {
                100.0 / TOP_N as f64
            },
            s2_token: hit.s2sphere_boundary.clone(),
        })
        .collect();

    InferenceResult {
        predictions,
        weighted_lat,
        weighted_lon,
    }
}

#[pyfunction]
fn infer_location(py: Python, vector: Vec<f32>) -> PyResult<InferenceResult> {
    if vector.len() != 64 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Expected 64d vector, got {}d", vector.len()),
        ));
    }

    py.allow_threads(|| {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Tokio error: {}", e)))?;

        let hits = rt
            .block_on(search_milvus(vector))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Milvus error: {}", e)))?;

        if hits.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "No results found",
            ));
        }

        Ok(weighted_interpolation(&hits))
    })
}

#[pyfunction]
fn infer_batch(py: Python, vectors: Vec<Vec<f32>>) -> PyResult<Vec<InferenceResult>> {
    let mut results = Vec::with_capacity(vectors.len());

    for vec in vectors {
        if vec.len() != 64 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Expected 64d vector, got {}d", vec.len()),
            ));
        }

        let result = py.allow_threads(|| {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Tokio error: {}", e)))?;

            let hits = rt
                .block_on(search_milvus(vec))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Milvus error: {}", e)))?;

            if hits.is_empty() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "No results found",
                ));
            }

            Ok(weighted_interpolation(&hits))
        })?;

        results.push(result);
    }

    Ok(results)
}

#[pymodule]
fn geoguessr_inference(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(infer_location, m)?)?;
    m.add_function(wrap_pyfunction!(infer_batch, m)?)?;
    m.add_class::<Prediction>()?;
    m.add_class::<InferenceResult>()?;
    Ok(())
}
