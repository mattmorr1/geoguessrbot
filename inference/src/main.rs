use clap::Parser;
use serde::{Deserialize, Serialize};
use std::io::Read;

const MILVUS_URL: &str = "http://localhost:19530";
const DB_NAME: &str = "geoguessr";
const COLLECTION: &str = "world_locations";
const TOP_K: usize = 10;
const TOP_N: usize = 3;
const EARTH_RADIUS_KM: f64 = 6371.0;

// --- CLI ---

#[derive(Parser)]
#[command(name = "inference", about = "AlphaEarth Scout inference engine")]
struct Cli {
    /// Path to a JSON file containing a 64d vector array. Reads from stdin if omitted.
    #[arg(short, long)]
    vector: Option<String>,

    /// Path to a JSON file containing a 32d hyperbolic tangent-space vector.
    /// When provided, runs a secondary L2 search on hyp_tangent_vec and fuses results.
    #[arg(long)]
    hyp_vector: Option<String>,

    /// Country code filter (ISO alpha-2, e.g. "US"). Constrains search to this country.
    #[arg(short, long)]
    country: Option<String>,

    /// S2 cell token (level 10) for sub-country filtering.
    #[arg(short, long)]
    s2_token: Option<String>,
}

// --- Output types ---

#[derive(Debug, Serialize)]
struct Prediction {
    lat: f64,
    lon: f64,
    confidence_pct: f64,
    s2_token: String,
    country_code: String,
    distance_km: f64,
}

#[derive(Debug, Serialize)]
struct InferenceResult {
    predictions: Vec<Prediction>,
    geodesic_lat: f64,
    geodesic_lon: f64,
    cluster_radius_km: f64,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    filter: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MilvusResponse {
    code: i32,
    data: Option<Vec<MilvusHit>>,
    message: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct MilvusHit {
    distance: f64,
    gps: GpsField,
    s2sphere_boundary: String,
    #[serde(default)]
    country_code: String,
}

#[derive(Debug, Clone, Deserialize)]
struct GpsField {
    lat: f64,
    lon: f64,
}

// --- Geodesic math ---

fn haversine(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (dlat, dlon) = (
        (lat2 - lat1).to_radians(),
        (lon2 - lon1).to_radians(),
    );
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    EARTH_RADIUS_KM * 2.0 * a.sqrt().atan2((1.0 - a).sqrt())
}

fn geodesic_center(hits: &[MilvusHit], weights: &[f64]) -> (f64, f64) {
    let total: f64 = weights.iter().sum();
    if total == 0.0 {
        return (hits[0].gps.lat, hits[0].gps.lon);
    }

    // weighted average in cartesian, then back to lat/lon
    let (mut x, mut y, mut z) = (0.0, 0.0, 0.0);
    for (hit, &w) in hits.iter().zip(weights.iter()) {
        let lat_r = hit.gps.lat.to_radians();
        let lon_r = hit.gps.lon.to_radians();
        x += w * lat_r.cos() * lon_r.cos();
        y += w * lat_r.cos() * lon_r.sin();
        z += w * lat_r.sin();
    }
    x /= total;
    y /= total;
    z /= total;

    let lat = z.atan2((x * x + y * y).sqrt()).to_degrees();
    let lon = y.atan2(x).to_degrees();
    (lat, lon)
}

// --- Milvus search ---

fn build_filter(country: &Option<String>, s2_token: &Option<String>) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(cc) = country {
        parts.push(format!("country_code == '{}'", cc));
    }
    if let Some(s2) = s2_token {
        parts.push(format!("s2_token_l10 == '{}'", s2));
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" AND "))
    }
}

async fn search_milvus(
    query_vec: Vec<f32>,
    anns_field: &str,
    filter: Option<String>,
) -> Result<Vec<MilvusHit>, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();

    let body = SearchRequest {
        db_name: DB_NAME.to_string(),
        collection_name: COLLECTION.to_string(),
        data: vec![query_vec],
        anns_field: anns_field.to_string(),
        limit: TOP_K,
        output_fields: vec![
            "gps".to_string(),
            "s2sphere_boundary".to_string(),
            "country_code".to_string(),
        ],
        filter,
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

// --- Refinement ---

fn refine(hits: &[MilvusHit]) -> InferenceResult {
    let weights: Vec<f64> = hits.iter().map(|h| h.distance.max(0.0)).collect();
    let total_weight: f64 = weights.iter().sum();

    let (geo_lat, geo_lon) = geodesic_center(hits, &weights);

    // cluster radius: max haversine distance from center to any hit
    let cluster_radius = hits
        .iter()
        .map(|h| haversine(geo_lat, geo_lon, h.gps.lat, h.gps.lon))
        .fold(0.0_f64, f64::max);

    let top_n = &hits[..TOP_N.min(hits.len())];
    let top_weights: Vec<f64> = top_n.iter().map(|h| h.distance.max(0.0)).collect();
    let top_sum: f64 = top_weights.iter().sum();

    let predictions: Vec<Prediction> = top_n
        .iter()
        .zip(top_weights.iter())
        .map(|(hit, &w)| {
            let dist = haversine(geo_lat, geo_lon, hit.gps.lat, hit.gps.lon);
            Prediction {
                lat: hit.gps.lat,
                lon: hit.gps.lon,
                confidence_pct: if top_sum > 0.0 {
                    (w / top_sum) * 100.0
                } else {
                    100.0 / TOP_N as f64
                },
                s2_token: hit.s2sphere_boundary.clone(),
                country_code: hit.country_code.clone(),
                distance_km: dist,
            }
        })
        .collect();

    InferenceResult {
        predictions,
        geodesic_lat: geo_lat,
        geodesic_lon: geo_lon,
        cluster_radius_km: cluster_radius,
    }
}

fn read_vector_file(path: &str, expected_dim: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let json_str = std::fs::read_to_string(path)?;
    let vec: Vec<f32> = serde_json::from_str(json_str.trim())?;
    if vec.len() != expected_dim {
        return Err(format!("Expected {}d vector, got {}d", expected_dim, vec.len()).into());
    }
    Ok(vec)
}

fn read_vector_stdin(expected_dim: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf)?;
    let vec: Vec<f32> = serde_json::from_str(buf.trim())?;
    if vec.len() != expected_dim {
        return Err(format!("Expected {}d vector, got {}d", expected_dim, vec.len()).into());
    }
    Ok(vec)
}

/// Fuse two hit lists by GPS proximity. If a DNA hit and a hyperbolic hit
/// are within `radius_km`, boost the DNA hit's weight. Otherwise keep both.
fn fuse_hits(dna_hits: Vec<MilvusHit>, hyp_hits: Vec<MilvusHit>, radius_km: f64) -> Vec<MilvusHit> {
    let mut fused = dna_hits;
    for hh in &hyp_hits {
        let already_close = fused.iter().any(|dh| {
            haversine(dh.gps.lat, dh.gps.lon, hh.gps.lat, hh.gps.lon) < radius_km
        });
        if already_close {
            // boost existing nearby hit
            for dh in fused.iter_mut() {
                if haversine(dh.gps.lat, dh.gps.lon, hh.gps.lat, hh.gps.lon) < radius_km {
                    dh.distance = (dh.distance + hh.distance) / 2.0;
                    break;
                }
            }
        } else {
            // novel location from hyperbolic search, add it
            fused.push(MilvusHit {
                distance: hh.distance * 0.5, // lower confidence for hyp-only hits
                gps: GpsField { lat: hh.gps.lat, lon: hh.gps.lon },
                s2sphere_boundary: hh.s2sphere_boundary.clone(),
                country_code: hh.country_code.clone(),
            });
        }
    }
    // re-sort by distance descending (higher = more similar for COSINE)
    fused.sort_by(|a, b| b.distance.partial_cmp(&a.distance).unwrap_or(std::cmp::Ordering::Equal));
    fused.truncate(TOP_K);
    fused
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let dna_vec = match &cli.vector {
        Some(path) => read_vector_file(path, 64),
        None => read_vector_stdin(64),
    };
    let dna_vec = match dna_vec {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error reading DNA vector: {e}");
            std::process::exit(1);
        }
    };

    let filter = build_filter(&cli.country, &cli.s2_token);
    if let Some(ref f) = filter {
        eprintln!("Constrained search: {f}");
    }

    // primary: COSINE search on alphaearth_vec
    let dna_hits = match search_milvus(dna_vec, "alphaearth_vec", filter.clone()).await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Error in DNA search: {e}");
            std::process::exit(1);
        }
    };

    // optional: L2 search on hyp_tangent_vec (Poincare proxy)
    let hits = if let Some(ref hyp_path) = cli.hyp_vector {
        let hyp_vec = match read_vector_file(hyp_path, 128) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Error reading hyperbolic vector: {e}");
                std::process::exit(1);
            }
        };
        eprintln!("Running dual search: DNA (COSINE) + Hyperbolic (L2)");
        let hyp_hits = match search_milvus(hyp_vec, "hyp_tangent_vec", filter).await {
            Ok(h) => h,
            Err(e) => {
                eprintln!("Hyperbolic search failed, falling back to DNA only: {e}");
                dna_hits.clone()
            }
        };
        fuse_hits(dna_hits, hyp_hits, 50.0)
    } else {
        dna_hits
    };

    if hits.is_empty() {
        eprintln!("No results found");
        std::process::exit(1);
    }

    let result = refine(&hits);
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
}
