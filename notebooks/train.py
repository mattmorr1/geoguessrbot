import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from pymilvus import MilvusClient
from tqdm import tqdm
from model import AlphaEarthScout, scout_loss, CurriculumScheduler, device

# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 32
ACCUM_STEPS = 32        # effective batch = 1024
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-2
CACHE_PATH = "training_cache.pt"
CHECKPOINT_DIR = "checkpoints"
MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = "root:Milvus"

USE_AMP = device == "cuda"
AMP_DTYPE = torch.float16 if USE_AMP else torch.float32

# ============================================================
# Country Code -> Continent Index
# ============================================================

_AF = dict.fromkeys([
    'DZ','AO','BJ','BW','BF','BI','CM','CV','CF','TD','KM','CG','CD','CI',
    'DJ','EG','GQ','ER','SZ','ET','GA','GM','GH','GN','GW','KE','LS','LR',
    'LY','MG','MW','ML','MR','MU','MA','MZ','NA','NE','NG','RW','ST','SN',
    'SC','SL','SO','ZA','SS','SD','TZ','TG','TN','UG','ZM','ZW','RE','YT',
], 0)

_AS = dict.fromkeys([
    'AF','AM','AZ','BH','BD','BT','BN','KH','CN','CY','GE','IN','ID','IR',
    'IQ','IL','JP','JO','KZ','KW','KG','LA','LB','MY','MV','MN','MM','NP',
    'KP','OM','PK','PS','PH','QA','SA','SG','KR','LK','SY','TW','TJ','TH',
    'TL','TR','TM','AE','UZ','VN','YE','HK','MO',
], 1)

_EU = dict.fromkeys([
    'AL','AD','AT','BY','BE','BA','BG','HR','CZ','DK','EE','FI','FR','DE',
    'GR','HU','IS','IE','IT','XK','LV','LI','LT','LU','MT','MD','MC','ME',
    'NL','MK','NO','PL','PT','RO','RU','SM','RS','SK','SI','ES','SE','CH',
    'UA','GB','VA','FO','GI','JE','GG','IM','AX',
], 2)

_NA = dict.fromkeys([
    'AG','BS','BB','BZ','CA','CR','CU','DM','DO','SV','GD','GT','HT','HN',
    'JM','MX','NI','PA','KN','LC','VC','TT','US','PR','VI','GL','BM','AW',
    'CW','SX','MQ','GP','TC','KY','MS','AI',
], 3)

_SA = dict.fromkeys([
    'AR','BO','BR','CL','CO','EC','GY','PY','PE','SR','UY','VE','GF','FK',
], 4)

_OC = dict.fromkeys([
    'AU','FJ','KI','MH','FM','NR','NZ','PW','PG','WS','SB','TO','TV','VU',
    'NC','PF','GU','AS','CK','NU','TK','WF','MP',
], 5)

_AQ = dict.fromkeys(['AQ'], 6)

CC_TO_CONTINENT = {**_AF, **_AS, **_EU, **_NA, **_SA, **_OC, **_AQ}
NUM_CONTINENTS = 7


# ============================================================
# Auxiliary Target Derivation
# ============================================================

def lat_to_climate_zone(lat):
    """Simplified Koppen-like climate zone from latitude.
    0=tropical, 1=arid, 2=temperate, 3=continental, 4=polar, 5=unknown
    """
    a = abs(lat)
    if a < 23.5:  return 0
    if a < 35.0:  return 1
    if a < 55.0:  return 2
    if a < 66.5:  return 3
    return 4

def coord_to_elevation_bin(lat, lon):
    """Rough elevation bin from geography. Uses latitude + longitude heuristics
    as a proxy until proper DEM data is integrated.
    Bins: 0=<0m, 1=0-100, 2=100-500, 3=500-1k, 4=1k-2k, 5=2k-3k, 6=3k-5k, 7=>5k

    This is intentionally coarse -- the auxiliary loss just needs a gradient
    signal to keep the DNA grounded. Replace with SRTM DEM lookup for accuracy.
    """
    a = abs(lat)
    # coastal / low-lying heuristic
    if a < 10 and abs(lon) > 20:
        return 1
    # high-altitude corridors: Andes, Himalayas, East Africa, Alps
    if 27 < lat < 40 and 70 < lon < 100:   return 5  # Himalaya/Tibet
    if -35 < lat < 5 and -80 < lon < -60:  return 4  # Andes
    if 43 < lat < 48 and 5 < lon < 16:     return 3  # Alps
    if -5 < lat < 15 and 28 < lon < 42:    return 3  # East African Rift
    # latitude-based fallback
    if a > 60:  return 1
    if a > 45:  return 2
    return 2


# ============================================================
# Data Export from Milvus
# ============================================================

def export_training_data():
    import reverse_geocoder as rg

    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN, db_name="geoguessr")
    client.load_collection("world_locations")

    all_data = []
    offset = 0
    page = 10000

    print("Pulling vectors from Milvus...")
    while True:
        try:
            batch = client.query(
                collection_name="world_locations",
                filter="id > 0",
                output_fields=["streetclip_vec", "alphaearth_vec", "gps"],
                limit=page,
                offset=offset,
            )
        except Exception as e:
            print(f"Query error at offset {offset}: {e}")
            break

        if not batch:
            break
        all_data.extend(batch)
        offset += page
        print(f"  {len(all_data)} records loaded...")

    print(f"Total: {len(all_data)} records")

    clip_vecs = torch.tensor([d['streetclip_vec'] for d in all_data], dtype=torch.float32)
    alpha_vecs = torch.tensor([d['alphaearth_vec'] for d in all_data], dtype=torch.float32)
    gps_coords = [(d['gps']['lat'], d['gps']['lon']) for d in all_data]

    print("Reverse geocoding GPS coordinates...")
    results = rg.search(gps_coords)
    country_codes = [r['cc'] for r in results]

    unique_countries = sorted(set(country_codes))
    cc_to_idx = {cc: i for i, cc in enumerate(unique_countries)}

    country_idx = torch.tensor([cc_to_idx[cc] for cc in country_codes], dtype=torch.long)
    continent_idx = torch.tensor(
        [CC_TO_CONTINENT.get(cc, 0) for cc in country_codes], dtype=torch.long
    )

    print("Deriving auxiliary targets (climate zones, elevation bins)...")
    climate_zones = torch.tensor(
        [lat_to_climate_zone(lat) for lat, lon in gps_coords], dtype=torch.long
    )
    elevation_bins = torch.tensor(
        [coord_to_elevation_bin(lat, lon) for lat, lon in gps_coords], dtype=torch.long
    )

    cache = {
        'clip_vecs': clip_vecs,
        'alpha_vecs': alpha_vecs,
        'country_idx': country_idx,
        'continent_idx': continent_idx,
        'climate_zones': climate_zones,
        'elevation_bins': elevation_bins,
        'cc_to_idx': cc_to_idx,
        'idx_to_cc': {v: k for k, v in cc_to_idx.items()},
        'unique_countries': unique_countries,
    }
    torch.save(cache, CACHE_PATH)
    print(f"Cached {len(all_data)} records -> {CACHE_PATH}")
    return cache


# ============================================================
# PyTorch Dataset
# ============================================================

class GeoDataset(Dataset):
    def __init__(self, clip_vecs, alpha_vecs, country_idx, continent_idx,
                 climate_zones, elevation_bins):
        self.clip = clip_vecs
        self.alpha = alpha_vecs
        self.country = country_idx
        self.continent = continent_idx
        self.climate = climate_zones
        self.elevation = elevation_bins

    def __len__(self):
        return len(self.clip)

    def __getitem__(self, i):
        return (self.clip[i], self.alpha[i], self.country[i],
                self.continent[i], self.climate[i], self.elevation[i])


# ============================================================
# Training Loop
# ============================================================

def train():
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached training data from {CACHE_PATH}")
        cache = torch.load(CACHE_PATH, weights_only=False)
    else:
        cache = export_training_data()

    num_countries = len(cache['unique_countries'])
    n_samples = len(cache['clip_vecs'])
    print(f"Countries: {num_countries} | Continents: {NUM_CONTINENTS} | Samples: {n_samples}")

    # handle old caches that lack auxiliary targets
    if 'climate_zones' not in cache:
        print("WARNING: cache missing auxiliary targets, regenerate with --force-export")
        print("         using placeholder zeros for climate/elevation")
        cache['climate_zones'] = torch.zeros(n_samples, dtype=torch.long)
        cache['elevation_bins'] = torch.zeros(n_samples, dtype=torch.long)

    model = AlphaEarthScout(num_countries=num_countries, num_continents=NUM_CONTINENTS).to(device)
    curriculum = CurriculumScheduler(total_epochs=EPOCHS, warmup=2)

    optimizer = torch.optim.AdamW(
        model.bridge.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler(enabled=USE_AMP)

    dataset = GeoDataset(
        cache['clip_vecs'], cache['alpha_vecs'],
        cache['country_idx'], cache['continent_idx'],
        cache['climate_zones'], cache['elevation_bins'],
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        w = curriculum.get_weights(epoch)
        print(f"\nEpoch {epoch + 1}/{EPOCHS} | weights: "
              f"a={w['alpha']:.2f} b={w['beta']:.2f} g={w['gamma']:.2f} x={w['aux']:.2f}")

        running_loss = 0
        running_parts = {'admin': 0, 'dna': 0, 'hier': 0, 'aux': 0}
        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"  training")
        for step, (clip_v, alpha_v, c_idx, cont_idx, clim, elev) in enumerate(pbar):
            clip_v = clip_v.to(device, non_blocking=True)
            alpha_v = alpha_v.to(device, non_blocking=True)
            c_idx = c_idx.to(device, non_blocking=True)
            cont_idx = cont_idx.to(device, non_blocking=True)
            clim = clim.to(device, non_blocking=True)
            elev = elev.to(device, non_blocking=True)

            with autocast(device_type='cuda', dtype=AMP_DTYPE, enabled=USE_AMP):
                outputs = model(clip_v)
                targets = {
                    'country_idx': c_idx,
                    'continent_idx': cont_idx,
                    'alpha_vecs': alpha_v,
                    'climate_zone': clim,
                    'elevation_bin': elev,
                }
                loss, parts = scout_loss(
                    outputs, targets, model.bridge.hierarchical, weights=w
                )
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.bridge.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * ACCUM_STEPS
            for k in running_parts:
                running_parts[k] += parts[k]

            if (step + 1) % 100 == 0:
                n = step + 1
                pbar.set_postfix({
                    'loss': f"{running_loss / n:.4f}",
                    'adm': f"{running_parts['admin'] / n:.3f}",
                    'dna': f"{running_parts['dna'] / n:.3f}",
                    'hyp': f"{running_parts['hier'] / n:.3f}",
                    'aux': f"{running_parts['aux'] / n:.3f}",
                })

        scheduler.step()

        avg_loss = running_loss / len(loader)
        lr = scheduler.get_last_lr()[0]
        print(f"  -> avg_loss={avg_loss:.4f} | lr={lr:.6f}")

        ckpt = {
            'epoch': epoch + 1,
            'bridge_state': model.bridge.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'num_countries': num_countries,
            'cc_to_idx': cache['cc_to_idx'],
            'idx_to_cc': cache['idx_to_cc'],
            'loss': avg_loss,
        }
        torch.save(ckpt, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch + 1}.pt"))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "best.pt"))
            print(f"  -> new best ({best_loss:.4f})")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    train()
