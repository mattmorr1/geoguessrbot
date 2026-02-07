use clap::Parser;
use serde::{Deserialize, Serialize};
use std::io::Read;

const MILVUS_URL: &str = "http://localhost:19530";
const DB_NAME: &str = "geoguessr";
const COLLECTION: &str = "world_locations";
const TOP_K: usize = 5;
const TOP_N: usize = 3;

// --- CLI ---

#[derive(Parser)]
#[command(name = "inference", about = "GeoguessrBot vector inference engine")]
struct Cli {
    /// Path to a JSON file containing a 64d vector array. Reads from stdin if omitted.
    #[arg(short, long)]
    vector: Option<String>,
}

// --- Output types ---

#[derive(Debug, Serialize)]
struct Prediction {
    lat: f64,
    lon: f64,
    confidence_pct: f64,
    s2_token: String,
}

#[derive(Debug, Serialize)]
struct InferenceResult {
    predictions: Vec<Prediction>,
    weighted_lat: f64,
    weighted_lon: f64,
}

// --- Milvus REST API types ---

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

// --- Core logic ---

async fn search_milvus(query_vec: Vec<f32>) -> Result<Vec<MilvusHit>, Box<dyn std::error::Error>> {
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

fn weighted_interpolation(hits: &[MilvusHit]) -> InferenceResult {
    // Milvus COSINE returns similarity scores (higher = more similar).
    // Use scores directly as weights.
    let weights: Vec<f64> = hits.iter().map(|h| h.distance.max(0.0)).collect();
    let total_weight: f64 = weights.iter().sum();

    // Weighted average GPS from all TOP_K results
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

    // Top-N predictions with confidence %
    // Hits are already sorted by score from Milvus (best first)
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

fn read_vector(cli: &Cli) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let json_str = match &cli.vector {
        Some(path) => std::fs::read_to_string(path)?,
        None => {
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf)?;
            buf
        }
    };

    let vec: Vec<f32> = serde_json::from_str(json_str.trim())?;

    if vec.len() != 64 {
        return Err(format!("Expected 64d vector, got {}d", vec.len()).into());
    }

    Ok(vec)
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let query_vec = match read_vector(&cli) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error reading vector: {e}");
            std::process::exit(1);
        }
    };

    let hits = match search_milvus(query_vec).await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Error searching Milvus: {e}");
            std::process::exit(1);
        }
    };

    if hits.is_empty() {
        eprintln!("No results found");
        std::process::exit(1);
    }

    let result = weighted_interpolation(&hits);
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
}
