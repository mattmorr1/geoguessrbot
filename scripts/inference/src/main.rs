use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Read;

const EARTH_RADIUS_KM: f64 = 6371.0088;
const TOP_OUT: usize = 5;

// --- CLI ---

#[derive(Parser)]
#[command(name = "inference", about = "AlphaEarth Scout serving: leaf-filtered Milvus search + product-of-experts rescoring")]
struct Cli {
    /// JSON-lines payload file from predict.py. Reads stdin if omitted.
    #[arg(short, long)]
    payload: Option<String>,

    #[arg(long, env = "MILVUS_URI", default_value = "http://localhost:19530")]
    milvus_uri: String,

    #[arg(long, env = "MILVUS_TOKEN", default_value = "root:Milvus")]
    milvus_token: String,

    #[arg(long, env = "MILVUS_DB", default_value = "geoguessr")]
    db: String,

    #[arg(long, env = "MILVUS_COLLECTION", default_value = "world_locations_v2")]
    collection: String,
}

// --- Payload / output ---

/// One query from predict.py: prior = log p(leaf|img) - log n_leaf, score = prior + kappa * cos.
#[derive(Debug, Deserialize)]
struct Payload {
    field: String,
    vec: Vec<f32>,
    cells: Vec<(i64, f64)>,
    kappa: f64,
    radius_km: f64,
    limit: usize,
    fallback: (f64, f64),
}

#[derive(Debug, Clone, Serialize)]
struct Scored {
    lat: f64,
    lon: f64,
    score: f64,
    sim: f64,
    leaf: i64,
    country_code: String,
}

#[derive(Debug, Serialize)]
struct Output {
    lat: f64,
    lon: f64,
    cluster_size: usize,
    cluster_radius_km: f64,
    fallback_used: bool,
    top: Vec<Scored>,
}

// --- Milvus REST API types ---

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SearchRequest<'a> {
    db_name: &'a str,
    collection_name: &'a str,
    data: [&'a [f32]; 1],
    anns_field: &'a str,
    limit: usize,
    output_fields: [&'static str; 4],
    filter: String,
}

#[derive(Debug, Deserialize)]
struct MilvusResponse {
    code: i32,
    data: Option<Vec<Hit>>,
    message: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct Hit {
    distance: f64,
    lat: f64,
    lon: f64,
    leaf: i64,
    #[serde(default)]
    country_code: String,
}

// --- Geodesic math ---

fn haversine(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (dlat, dlon) = ((lat2 - lat1).to_radians(), (lon2 - lon1).to_radians());
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    EARTH_RADIUS_KM * 2.0 * a.sqrt().atan2((1.0 - a).sqrt())
}

/// Weighted mean of unit vectors, projected back to the sphere.
fn geodesic_center(points: &[(f64, f64)], weights: &[f64]) -> (f64, f64) {
    let (x, y, z) = points.iter().zip(weights).fold((0.0, 0.0, 0.0), |(x, y, z), (&(lat, lon), &w)| {
        let (la, lo) = (lat.to_radians(), lon.to_radians());
        (x + w * la.cos() * lo.cos(), y + w * la.cos() * lo.sin(), z + w * la.sin())
    });
    if x == 0.0 && y == 0.0 && z == 0.0 {
        return points[0];
    }
    (z.atan2((x * x + y * y).sqrt()).to_degrees(), y.atan2(x).to_degrees())
}

// --- Scoring (mirrors eval.py serve mode) ---

fn build_filter(cells: &[(i64, f64)]) -> Option<String> {
    if cells.is_empty() {
        return None;
    }
    let ids: Vec<String> = cells.iter().map(|(l, _)| l.to_string()).collect();
    Some(format!("leaf in [{}]", ids.join(",")))
}

fn rescore(hits: Vec<Hit>, cells: &[(i64, f64)], kappa: f64) -> Vec<Scored> {
    let prior: HashMap<i64, f64> = cells.iter().copied().collect();
    let mut scored: Vec<Scored> = hits
        .into_iter()
        .filter_map(|h| {
            prior.get(&h.leaf).map(|p| Scored {
                lat: h.lat,
                lon: h.lon,
                score: p + kappa * h.distance,
                sim: h.distance,
                leaf: h.leaf,
                country_code: h.country_code,
            })
        })
        .collect();
    scored.sort_by(|a, b| b.score.total_cmp(&a.score));
    scored
}

/// Score-weighted center of hits within radius_km of the best hit; None when there are no hits.
fn refine(scored: &[Scored], radius_km: f64) -> Option<(f64, f64, usize, f64)> {
    let best = scored.first()?;
    let cluster: Vec<&Scored> = scored
        .iter()
        .filter(|s| haversine(best.lat, best.lon, s.lat, s.lon) <= radius_km)
        .collect();
    let points: Vec<(f64, f64)> = cluster.iter().map(|s| (s.lat, s.lon)).collect();
    let weights: Vec<f64> = cluster.iter().map(|s| (s.score - best.score).exp()).collect();
    let (lat, lon) = geodesic_center(&points, &weights);
    let radius = points.iter().map(|&(a, b)| haversine(lat, lon, a, b)).fold(0.0_f64, f64::max);
    Some((lat, lon, cluster.len(), radius))
}

fn resolve(hits: Vec<Hit>, p: &Payload) -> Output {
    let scored = rescore(hits, &p.cells, p.kappa);
    match refine(&scored, p.radius_km) {
        Some((lat, lon, n, r)) => Output {
            lat,
            lon,
            cluster_size: n,
            cluster_radius_km: r,
            fallback_used: false,
            top: scored.into_iter().take(TOP_OUT).collect(),
        },
        None => Output {
            lat: p.fallback.0,
            lon: p.fallback.1,
            cluster_size: 0,
            cluster_radius_km: 0.0,
            fallback_used: true,
            top: Vec::new(),
        },
    }
}

// --- Milvus search ---

async fn search(client: &reqwest::Client, cli: &Cli, p: &Payload) -> Result<Vec<Hit>, Box<dyn std::error::Error>> {
    let Some(filter) = build_filter(&p.cells) else {
        return Ok(Vec::new());
    };
    let body = SearchRequest {
        db_name: &cli.db,
        collection_name: &cli.collection,
        data: [&p.vec],
        anns_field: &p.field,
        limit: p.limit,
        output_fields: ["lat", "lon", "leaf", "country_code"],
        filter,
    };
    let resp: MilvusResponse = client
        .post(format!("{}/v2/vectordb/entities/search", cli.milvus_uri))
        .bearer_auth(&cli.milvus_token)
        .json(&body)
        .send()
        .await?
        .json()
        .await?;
    if resp.code != 0 {
        return Err(format!("Milvus error (code {}): {}", resp.code, resp.message.unwrap_or_default()).into());
    }
    Ok(resp.data.unwrap_or_default())
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let input = match &cli.payload {
        Some(path) => std::fs::read_to_string(path),
        None => {
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf).map(|_| buf)
        }
    };
    let input = input.unwrap_or_else(|e| {
        eprintln!("Error reading payload: {e}");
        std::process::exit(1);
    });

    let client = reqwest::Client::new();
    for line in input.lines().filter(|l| !l.trim().is_empty()) {
        let payload: Payload = serde_json::from_str(line).unwrap_or_else(|e| {
            eprintln!("Invalid payload: {e}");
            std::process::exit(1);
        });
        let hits = search(&client, &cli, &payload).await.unwrap_or_else(|e| {
            eprintln!("Search failed: {e}");
            std::process::exit(1);
        });
        println!("{}", serde_json::to_string(&resolve(hits, &payload)).unwrap());
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(lat: f64, lon: f64, sim: f64, leaf: i64) -> Hit {
        Hit { distance: sim, lat, lon, leaf, country_code: String::new() }
    }

    fn payload(cells: Vec<(i64, f64)>, radius_km: f64) -> Payload {
        Payload { field: "ae_vec".into(), vec: vec![0.0; 64], cells, kappa: 10.0, radius_km, limit: 64, fallback: (1.0, 2.0) }
    }

    #[test]
    fn test_haversine_paris_london() {
        // Paris (48.8566, 2.3522) → London (51.5074, -0.1278) ≈ 341 km
        let d = haversine(48.8566, 2.3522, 51.5074, -0.1278);
        assert!((d - 341.0).abs() < 5.0, "expected ~341 km, got {d:.1}");
    }

    #[test]
    fn test_haversine_same_point() {
        let d = haversine(48.8566, 2.3522, 48.8566, 2.3522);
        assert!(d < 1e-6, "same-point distance should be ~0, got {d}");
    }

    #[test]
    fn test_haversine_antipodal() {
        // Antipodal points should be ~π * R ≈ 20015 km
        let d = haversine(0.0, 0.0, 0.0, 180.0);
        assert!((d - 20015.0).abs() < 10.0, "antipodal distance: {d:.1}");
    }

    #[test]
    fn test_geodesic_center_symmetric() {
        let (lat, lon) = geodesic_center(&[(10.0, 0.0), (-10.0, 0.0)], &[1.0, 1.0]);
        assert!(lat.abs() < 1e-6 && lon.abs() < 1e-6, "centre should be (0,0), got ({lat},{lon})");
    }

    #[test]
    fn test_geodesic_center_single_point() {
        let (lat, lon) = geodesic_center(&[(48.8566, 2.3522)], &[0.9]);
        assert!((lat - 48.8566).abs() < 1e-4 && (lon - 2.3522).abs() < 1e-4);
    }

    #[test]
    fn test_build_filter() {
        assert_eq!(build_filter(&[]), None);
        assert_eq!(build_filter(&[(3, -1.0), (17, -2.0)]).unwrap(), "leaf in [3,17]");
    }

    #[test]
    fn test_rescore_prior_outweighs_small_similarity_gap() {
        // leaf 1 has a much stronger cell prior; a slightly higher cosine in leaf 2 must not win
        let hits = vec![hit(0.0, 0.0, 0.80, 2), hit(10.0, 10.0, 0.78, 1)];
        let s = rescore(hits, &[(1, -0.5), (2, -5.0)], 10.0);
        assert_eq!(s[0].leaf, 1);
        assert!((s[0].score - (-0.5 + 7.8)).abs() < 1e-9);
    }

    #[test]
    fn test_rescore_drops_hits_outside_payload_cells() {
        let s = rescore(vec![hit(0.0, 0.0, 0.9, 99)], &[(1, 0.0)], 10.0);
        assert!(s.is_empty());
    }

    #[test]
    fn test_refine_excludes_far_outlier() {
        // two Paris hits and one NYC hit: a 25 km cluster must stay in Paris, not drift into the Atlantic
        let hits = vec![hit(48.8566, 2.3522, 0.9, 1), hit(48.86, 2.36, 0.88, 1), hit(40.7128, -74.006, 0.89, 1)];
        let out = resolve(hits, &payload(vec![(1, 0.0)], 25.0));
        assert_eq!(out.cluster_size, 2);
        assert!(haversine(out.lat, out.lon, 48.8566, 2.3522) < 2.0);
        assert!(!out.fallback_used);
    }

    #[test]
    fn test_resolve_falls_back_without_hits() {
        let out = resolve(Vec::new(), &payload(vec![(1, 0.0)], 25.0));
        assert!(out.fallback_used);
        assert_eq!((out.lat, out.lon), (1.0, 2.0));
    }

    #[test]
    fn test_payload_parses_predict_py_shape() {
        let p: Payload = serde_json::from_str(
            r#"{"field":"ae_vec","vec":[0.1,0.2],"cells":[[5,-1.25]],"kappa":20,"radius_km":25,"limit":64,"fallback":[48.8,2.3]}"#,
        )
        .unwrap();
        assert_eq!(p.cells, vec![(5, -1.25)]);
        assert_eq!(p.fallback, (48.8, 2.3));
    }
}
