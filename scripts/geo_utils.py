import time
import s2sphere
import ee

def get_s2_token(lat, lon, level=12):
    p = s2sphere.LatLng.from_degrees(lat, lon)
    return s2sphere.CellId.from_lat_lng(p).parent(level).to_token()

def get_streetclip_embedding(images, processor, model, device):
    import torch
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return features.cpu().numpy()

def alpha_earth_features(coords_list, max_retries=5):
    for attempt in range(max_retries):
        try:
            points = [ee.Feature(ee.Geometry.Point([lon, lat])) for lat, lon in coords_list]
            fc = ee.FeatureCollection(points)
            ae_img = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").first()
            sampled = ae_img.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=10)
            results = []
            for feat in sampled.getInfo()['features']:
                props = feat['properties']
                vec = [float(props.get(f'A{i:02d}', 0.0) or 0.0) for i in range(64)]
                results.append(vec)
            return results
        except Exception as e:
            if attempt < max_retries - 1 and any(x in str(e).lower() for x in ['resolve', 'connection', 'retries', 'timeout']):
                delay = 2 * (2 ** attempt)
                print(f"  [!] GEE retry {attempt+1}/{max_retries} in {delay}s...")
                time.sleep(delay)
            else:
                raise
