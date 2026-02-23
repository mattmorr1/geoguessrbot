import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.amp import autocast, GradScaler
from pymilvus import MilvusClient
from tqdm import tqdm
from model import (
    AlphaEarthScout, ContrastiveQueue, CurriculumScheduler,
    scout_loss, device,
)

# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 32
ACCUM_STEPS = 32
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-2
WARMUP_EPOCHS = 2
EMA_DECAY = 0.999
QUEUE_SIZE = 8192
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
    a = abs(lat)
    if a < 23.5:  return 0
    if a < 35.0:  return 1
    if a < 55.0:  return 2
    if a < 66.5:  return 3
    return 4

def coord_to_elevation_bin(lat, lon):
    a = abs(lat)
    if a < 10 and abs(lon) > 20:       return 1
    if 27 < lat < 40 and 70 < lon < 100:   return 5
    if -35 < lat < 5 and -80 < lon < -60:  return 4
    if 43 < lat < 48 and 5 < lon < 16:     return 3
    if -5 < lat < 15 and 28 < lon < 42:    return 3
    if a > 60:  return 1
    if a > 45:  return 2
    return 2


# ============================================================
# EMA (Exponential Moving Average)
# ============================================================
# Maintains a shadow copy of model weights that's a running average.
# Produces smoother, more generalizable weights than the final
# training iterate. Used for the deployed model.

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def state_dict(self):
        return dict(self.shadow)

    def apply(self, model):
        model.load_state_dict(self.shadow)


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

    print("Reverse geocoding...")
    results = rg.search(gps_coords)
    country_codes = [r['cc'] for r in results]

    unique_countries = sorted(set(country_codes))
    cc_to_idx = {cc: i for i, cc in enumerate(unique_countries)}

    country_idx = torch.tensor([cc_to_idx[cc] for cc in country_codes], dtype=torch.long)
    continent_idx = torch.tensor(
        [CC_TO_CONTINENT.get(cc, 0) for cc in country_codes], dtype=torch.long
    )

    print("Deriving auxiliary targets...")
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
        print(f"Loading cache: {CACHE_PATH}")
        cache = torch.load(CACHE_PATH, weights_only=False)
    else:
        cache = export_training_data()

    num_countries = len(cache['unique_countries'])
    n_samples = len(cache['clip_vecs'])
    print(f"Countries: {num_countries} | Samples: {n_samples}")

    if 'climate_zones' not in cache:
        print("WARNING: cache missing aux targets, using zeros")
        cache['climate_zones'] = torch.zeros(n_samples, dtype=torch.long)
        cache['elevation_bins'] = torch.zeros(n_samples, dtype=torch.long)

    model = AlphaEarthScout(num_countries=num_countries, num_continents=NUM_CONTINENTS).to(device)

    # torch.compile on CUDA for fused kernels
    if device == 'cuda' and hasattr(torch, 'compile'):
        model.bridge = torch.compile(model.bridge, mode='reduce-overhead')
        print("Bridge compiled with torch.compile")

    curriculum = CurriculumScheduler(total_epochs=EPOCHS, warmup=WARMUP_EPOCHS)
    queue = ContrastiveQueue(dim=64, size=QUEUE_SIZE)
    ema = EMA(model.bridge, decay=EMA_DECAY)

    optimizer = torch.optim.AdamW(
        model.bridge.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    # country-balanced sampling: inverse frequency weighting
    counts = torch.bincount(cache['country_idx'], minlength=num_countries).float()
    sample_weights = 1.0 / counts[cache['country_idx']]
    sampler = WeightedRandomSampler(sample_weights, num_samples=n_samples, replacement=True)

    dataset = GeoDataset(
        cache['clip_vecs'], cache['alpha_vecs'],
        cache['country_idx'], cache['continent_idx'],
        cache['climate_zones'], cache['elevation_bins'],
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    # LR schedule: linear warmup then cosine decay (step-level)
    steps_per_epoch = len(loader)
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    total_steps = EPOCHS * steps_per_epoch
    warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_steps])

    scaler = GradScaler(enabled=USE_AMP)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        w = curriculum.get_weights(epoch)
        print(f"\nEpoch {epoch + 1}/{EPOCHS} | "
              f"a={w['alpha']:.2f} b={w['beta']:.2f} g={w['gamma']:.2f} x={w['aux']:.2f}")

        running_loss = 0
        running_parts = {'admin': 0, 'dna': 0, 'hier': 0, 'aux': 0}
        optimizer.zero_grad()

        pbar = tqdm(loader, desc="  training")
        for step, (clip_v, alpha_v, c_idx, cont_idx, clim, elev) in enumerate(pbar):
            clip_v = clip_v.to(device, non_blocking=True)
            alpha_v = alpha_v.to(device, non_blocking=True)
            c_idx = c_idx.to(device, non_blocking=True)
            cont_idx = cont_idx.to(device, non_blocking=True)
            clim = clim.to(device, non_blocking=True)
            elev = elev.to(device, non_blocking=True)

            queue_negs = queue.get(device)

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
                    outputs, targets, model.bridge.hierarchical,
                    temperature=model.bridge.temperature,
                    weights=w, queue_negs=queue_negs,
                )
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            # enqueue current batch targets for future negatives
            queue.push(alpha_v)

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.bridge.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model.bridge)

            scheduler.step()

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
                    't': f"{parts['temp']:.4f}",
                })

        avg_loss = running_loss / len(loader)
        lr = optimizer.param_groups[0]['lr']
        print(f"  -> loss={avg_loss:.4f} lr={lr:.6f} temp={model.bridge.temperature.item():.4f}")

        ckpt = {
            'epoch': epoch + 1,
            'bridge_state': model.bridge.state_dict(),
            'ema_state': ema.state_dict(),
            'optimizer_state': optimizer.state_dict(),
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


# ============================================================
# Post-Training: Backfill hyp_tangent_vec into Milvus
# ============================================================

def backfill_hyp_tangent(checkpoint_path="checkpoints/best.pt", inference_batch=512):
    """After training, compute and upsert hyp_tangent vectors for all records.
    Run this once after training completes so the Rust inference engine
    can use the hyperbolic tangent search path.

    Upsert is delete+insert in Milvus, so we must fetch ALL fields per record
    to avoid destroying existing data.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False)

    num_countries = ckpt['num_countries']
    model = AlphaEarthScout(num_countries=num_countries, num_continents=NUM_CONTINENTS).to(device)

    # use EMA weights for inference
    if 'ema_state' in ckpt:
        model.bridge.load_state_dict(ckpt['ema_state'])
        print("Loaded EMA weights")
    else:
        model.bridge.load_state_dict(ckpt['bridge_state'])
    model.eval()

    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN, db_name="geoguessr")
    client.load_collection("world_locations")

    ALL_FIELDS = [
        "id", "streetclip_vec", "alphaearth_vec", "hyp_tangent_vec",
        "gps", "s2sphere_boundary", "country_code", "continent", "s2_token_l10",
    ]

    offset = 0
    page = 5000
    total_updated = 0

    print("Backfilling hyp_tangent vectors...")
    while True:
        records = client.query(
            collection_name="world_locations",
            filter="id > 0",
            output_fields=ALL_FIELDS,
            limit=page,
            offset=offset,
        )
        if not records:
            break

        clip_vecs = torch.tensor(
            [r['streetclip_vec'] for r in records], dtype=torch.float32
        )

        # sub-batch inference to avoid OOM on large pages
        all_tangent = []
        for i in range(0, len(clip_vecs), inference_batch):
            batch = clip_vecs[i:i + inference_batch].to(device)
            with torch.no_grad():
                out = model(batch)
            all_tangent.append(out['hyp_tangent'].cpu())
        tangent_vecs = torch.cat(all_tangent).numpy().tolist()

        # rebuild full records with the new tangent vectors
        upsert_data = []
        for rec, tvec in zip(records, tangent_vecs):
            rec["hyp_tangent_vec"] = tvec
            upsert_data.append(rec)

        try:
            client.upsert(collection_name="world_locations", data=upsert_data)
            total_updated += len(upsert_data)
            print(f"  {total_updated} records updated...")
        except Exception as e:
            print(f"  Upsert failed at offset {offset}: {e}")

        offset += page

    print(f"Backfill complete: {total_updated} records updated")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "backfill":
        ckpt_path = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/best.pt"
        backfill_hyp_tangent(ckpt_path)
    else:
        train()
