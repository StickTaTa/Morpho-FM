"""
Unified multi-seed ablation training for the Morpho-FM decoder-initialization study.

One script, one run:
  EXP_MODE   single | multi     training regime
  ARM        pretrained | random   decoder initialization
  MODEL_SEED int                model init + data-shuffle seed
  SPLIT_SEED int                blocked within-slide validation split seed

Both arms run this same file with the same hyper-parameters; `ARM` selects only
whether the CellFM checkpoint is loaded. That is the whole point: the published
single-seed comparison had the two arms trained by two different scripts, which had
drifted apart (batch_size, lr, two different `make_blocked_split` implementations, and
a validation split of 29.99% vs 20.25% of INT25). Those deltas are recorded in
`docs/tmp_methods_core_code_detail_20260329.md` versus the executed server scripts; this
file removes them by construction.

Relative to the historical scripts, two things are deliberate:

  SPLIT_SEED is decoupled from MODEL_SEED and pinned, so every run validates on the
  identical spatial partition and the only varying factor is initialization. The
  historical coupling is what produced the 29.99/20.25 mismatch, because
  `make_blocked_split` was called with the model seed.

  The validation curve is written to `train_log.csv` and `best_epoch`/`epochs_run` are
  recorded. The historical runs saved only `best_model.pt`, so there was no way to show
  where training plateaued.

Set PROBE_STEPS>0 to benchmark throughput instead of training: the script runs that
many gene-chunk steps, prints timing JSON and exits. Used to choose the Slurm
granularity (runs per node) from measurement rather than assumption.

Run under torchrun:
  torchrun --standalone --nproc_per_node=4 08_st_superres_train_cellfm_ablation_multiseed.py
"""
from pathlib import Path
import os
import sys
import json
import random
import time

import numpy as np
import torch

ROOT = Path(os.environ.get("MORPHO_VC_ROOT", "..")).expanduser().resolve()
os.chdir(ROOT)
sys.path.append(str(ROOT / "src"))
print(f"ROOT: {ROOT}")

# ----------------------------
# 0) Experiment config (environment-driven)
# ----------------------------
EXP_MODE = os.environ.get("EXP_MODE", "single")
ARM = os.environ.get("ARM", "random")
MODEL_SEED = int(os.environ.get("MODEL_SEED", "42"))
SPLIT_SEED = int(os.environ.get("SPLIT_SEED", "42"))
PROBE_STEPS = int(os.environ.get("PROBE_STEPS", "0"))

if EXP_MODE not in {"multi", "single"}:
    raise ValueError("EXP_MODE must be 'multi' or 'single'")
if ARM not in {"pretrained", "random"}:
    raise ValueError("ARM must be 'pretrained' or 'random'")

if EXP_MODE == "multi":
    train_ids = ["INT25", "INT26"]
    val_ids = ["INT27"]
    test_id = "INT28"
    common_gene_slides = train_ids + val_ids
    cache_slides = train_ids + val_ids
else:
    train_ids = ["INT25"]
    val_ids = []
    test_id = "INT28"
    train_slide_id = "INT25"
    common_gene_slides = ["INT25", "INT26", "INT27", "INT28"]
    cache_slides = train_ids

RUN_NAME = f"{EXP_MODE}_{ARM}_seed{MODEL_SEED}" + os.environ.get("RUN_SUFFIX", "")
CKPT_DIR = ROOT / "checkpoints" / "st_superres" / "ablation_multiseed" / RUN_NAME
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# The single variable that defines the ablation.
cellfm_checkpoint = (
    ROOT / "checkpoints" / "CellFM" / "CellFM_80M_weight.pt" if ARM == "pretrained" else None
)

gene_vocab_path = ROOT / "assets" / "cellfm" / "expand_gene_info.csv"

print("EXP_MODE:", EXP_MODE, "| ARM:", ARM, "| MODEL_SEED:", MODEL_SEED, "| SPLIT_SEED:", SPLIT_SEED)
print("Train IDs:", train_ids, "| Val IDs:", val_ids, "| Test ID:", test_id)
print("CKPT_DIR:", CKPT_DIR)
print("cellfm_checkpoint:", cellfm_checkpoint)


# %%
# ----------------------------
# 1) Build / refresh feature cache + genes
# ----------------------------
from st_pipeline.superres.feature_cache import ensure_feature_cache, get_common_genes
from st_pipeline.data.gene_vocab import load_gene_vocab, map_genes_to_vocab

force_recompute = False
max_dim = 16000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ensure_feature_cache(
    slide_ids=cache_slides,
    root=ROOT,
    force=force_recompute,
    device=device,
    max_dim=max_dim,
    use_tissue_seg=True,
)

common_genes = get_common_genes(common_gene_slides, root=ROOT)
vocab = load_gene_vocab(gene_vocab_path)
kept_genes, gene_ids = map_genes_to_vocab(common_genes, vocab)

(CKPT_DIR / "common_genes.json").write_text(json.dumps(kept_genes, indent=2), encoding="utf-8")
print(f"Common genes: raw={len(common_genes)} | mapped={len(kept_genes)}")


# %%
# ----------------------------
# 2) Dataset (same as the 08 pipeline)
# ----------------------------
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from st_pipeline.superres.feature_cache import load_cache_slide, normalize_slide
from st_pipeline.constants import KEYS


def get_disk_mask(radius):
    radius_ceil = int(np.ceil(radius))
    locs = np.meshgrid(
        np.arange(-radius_ceil, radius_ceil + 1),
        np.arange(-radius_ceil, radius_ceil + 1),
        indexing="ij",
    )
    locs = np.stack(locs, axis=-1)
    distsq = (locs ** 2).sum(-1)
    return distsq <= radius ** 2


GRID_FACTOR = 16


class SuperresSpotDataset(Dataset):
    def __init__(self, embs, cnts, locs_px, radius_px, gene_ids, grid_factor=GRID_FACTOR):
        super().__init__()
        self.embs = embs
        self.cnts = cnts
        self.gene_ids = gene_ids

        locs_px = np.asarray(locs_px, dtype=np.float32)
        self.locs_px = locs_px.copy()
        locs_grid = np.round(locs_px / float(grid_factor)).astype(np.int64)

        self.radius = float(radius_px) / float(grid_factor)
        self.mask = get_disk_mask(self.radius)
        self.mask_size = int(self.mask.sum())

        H, W = embs.shape[:2]
        r0, r1 = self._patch_offsets()
        valid = (
            (locs_grid[:, 0] + r0[0] >= 0) &
            (locs_grid[:, 0] + r0[1] <= H) &
            (locs_grid[:, 1] + r1[0] >= 0) &
            (locs_grid[:, 1] + r1[1] <= W)
        )
        finite_center = np.isfinite(embs[locs_grid[:, 0], locs_grid[:, 1]]).all(-1)
        valid &= finite_center

        self.keep_idx = np.where(valid)[0]
        self.locs = locs_grid[valid]
        self.locs_px = self.locs_px[valid]
        self.cnts = cnts[valid]

        sums = self.cnts.sum(axis=1)
        self.size_factors = (sums / (sums.mean() + 1e-8)).astype(np.float32)

    def _patch_offsets(self):
        shape = np.array(self.mask.shape)
        center = shape // 2
        r = np.stack([-center, shape - center], -1)
        return r[0], r[1]

    def __len__(self):
        return len(self.locs)

    def __getitem__(self, idx):
        i, j = self.locs[idx]
        r0, r1 = self._patch_offsets()
        patch = self.embs[i + r0[0]: i + r0[1], j + r1[0]: j + r1[1]]
        x = patch if self.mask.all() else patch[self.mask]
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(self.cnts[idx].astype(np.float32))
        return {
            KEYS.X: x,
            KEYS.Y_BAG: y,
            KEYS.SIZE_FACTOR: float(self.size_factors[idx]),
            KEYS.GENE_IDS: self.gene_ids,
        }


def _compute_blocks(coords, grid_size):
    min_x, min_y = coords.min(0)
    max_x, max_y = coords.max(0)
    spans = np.array([max_x - min_x + 1e-6, max_y - min_y + 1e-6])
    step = spans / grid_size
    blocks = {}
    for i, (x, y) in enumerate(coords):
        gx = int((x - min_x) / step[0])
        gy = int((y - min_y) / step[1])
        gx = min(max(gx, 0), grid_size - 1)
        gy = min(max(gy, 0), grid_size - 1)
        blocks.setdefault((gx, gy), []).append(i)
    return blocks


def _select_grid_size(coords, min_grid=4, max_grid=10, min_blocks=20, max_blocks=80):
    target = (min_blocks + max_blocks) / 2.0
    candidates = []
    for g in range(min_grid, max_grid + 1):
        blocks = _compute_blocks(coords, g)
        n_blocks = len(blocks)
        in_range = min_blocks <= n_blocks <= max_blocks
        score = 0 if in_range else abs(n_blocks - target)
        candidates.append((in_range, score, g, blocks, n_blocks))
    candidates.sort(key=lambda x: (not x[0], x[1]))
    in_range, score, grid_size, blocks, n_blocks = candidates[0]
    return grid_size, blocks, n_blocks


def make_blocked_split(coords, val_ratio=0.2, seed=42, grid_size=None,
                       min_blocks=20, max_blocks=80):
    """Quadrant-balanced blocked-grid split.

    This is the implementation documented in
    `docs/tmp_methods_core_code_detail_20260329.md` (section on the split), taken from
    `08_st_superres_train_cellfm_single_INT25.py`, not the simpler greedy variant that
    the historical random-init ablation script used.
    """
    rng = random.Random(seed)
    coords = np.asarray(coords, dtype=np.float32)
    n_spots = coords.shape[0]

    if grid_size is None:
        grid_size, blocks, block_count = _select_grid_size(
            coords, min_blocks=min_blocks, max_blocks=max_blocks
        )
    else:
        blocks = _compute_blocks(coords, grid_size)
        block_count = len(blocks)

    if block_count < 2:
        raise ValueError("Not enough blocks to create a validation split.")

    block_sizes = {bid: len(idxs) for bid, idxs in blocks.items()}

    quadrants = {0: [], 1: [], 2: [], 3: []}
    for bid in blocks:
        gx, gy = bid
        q = (gx >= grid_size / 2) * 1 + (gy >= grid_size / 2) * 2
        quadrants[q].append(bid)

    val_blocks = set()
    target_total = max(1, int(round(n_spots * val_ratio)))

    for q in range(4):
        block_ids = quadrants[q]
        if len(block_ids) < 2:
            continue
        rng_q = random.Random(seed + q + 1)
        block_ids = block_ids.copy()
        rng_q.shuffle(block_ids)
        total_q = sum(block_sizes[bid] for bid in block_ids)
        target_q = int(round(total_q * val_ratio))
        picked = 0
        selected_q = []
        for bid in block_ids:
            if len(selected_q) >= len(block_ids) - 1:
                break
            picked += block_sizes[bid]
            selected_q.append(bid)
            if picked >= target_q:
                break
        val_blocks.update(selected_q)

    val_count = sum(block_sizes[bid] for bid in val_blocks)
    if val_count < target_total:
        remaining = [bid for bid in blocks.keys() if bid not in val_blocks]
        remaining.sort(key=lambda b: block_sizes[b], reverse=True)
        for bid in remaining:
            if len(val_blocks) >= len(blocks) - 1:
                break
            val_blocks.add(bid)
            val_count += block_sizes[bid]
            if val_count >= target_total:
                break

    val_indices = []
    for bid in val_blocks:
        val_indices.extend(blocks[bid])
    val_indices = sorted(set(val_indices))
    val_set = set(val_indices)
    train_indices = [i for i in range(n_spots) if i not in val_set]

    actual_ratio = len(val_indices) / max(n_spots, 1)
    meta = {
        "strategy": "blocked_grid_quadrant",
        "grid_size": grid_size,
        "block_count": block_count,
        "val_ratio_target": val_ratio,
        "val_ratio_actual": actual_ratio,
        "split_seed": seed,
    }
    return train_indices, val_indices, meta


slide_stats = {}
datasets = {}
for sid in cache_slides:
    embs, cnts, locs, radius, _, _ = load_cache_slide(sid, root=ROOT, genes=kept_genes)
    embs_norm, _, stats = normalize_slide(embs, cnts)
    ds = SuperresSpotDataset(embs_norm, cnts, locs, radius, gene_ids)
    datasets[sid] = ds
    slide_stats[sid] = {
        "radius_px": float(radius),
        "mask_size": int(ds.mask_size),
        "embs_mean": stats["embs_mean"],
        "embs_std": stats["embs_std"],
    }
    print(f"{sid}: embs={embs.shape} cnts={cnts.shape} spots={len(ds)}")

for sid, st in slide_stats.items():
    np.savez(CKPT_DIR / f"stats_{sid}.npz", **st)

if EXP_MODE == "multi":
    train_ds = ConcatDataset([datasets[sid] for sid in train_ids])
    val_ds = ConcatDataset([datasets[sid] for sid in val_ids])
    split_meta = None
else:
    base_ds = datasets[train_slide_id]
    # SPLIT_SEED, not MODEL_SEED: the validation partition is identical for every run.
    tr_idx, va_idx, split_meta = make_blocked_split(
        base_ds.locs_px, val_ratio=0.2, seed=SPLIT_SEED
    )
    train_ds = Subset(base_ds, tr_idx)
    val_ds = Subset(base_ds, va_idx)
    (CKPT_DIR / "single_blocked_split.json").write_text(
        json.dumps(split_meta, indent=2), encoding="utf-8"
    )
    print("Single split:", split_meta)

print(f"Train spots: {len(train_ds)} | Val spots: {len(val_ds)}")


# %%
# ----------------------------
# 3) Train CellFM model (DDP-ready)
# ----------------------------
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from st_pipeline.data.collate import mil_collate
from st_pipeline.model.morpho_cellfm_mil import MorphoCellfmMIL
from st_pipeline.model.nb_loss import nb_negative_log_likelihood

# Frozen across both arms. lr=1e-4 and batch_size=12 match the values the historical
# runs used; `docs/tmp_methods_core_code_detail_20260329.md` records batch_size=4 for
# the notebooks, but the executed server scripts used 12.
batch_size = 12
num_workers = 4
epochs = int(os.environ.get("MAX_EPOCHS", "25"))
lr = 1e-4
gene_chunk_size = 256
gene_lr_mult = 3.0
heartbeat_batches = 5
use_amp = True
early_stop_patience = 5
# 1e-4, the historical value. Measured on this pipeline it does not actually fire: after
# roughly epoch 20 the validation loss still falls by 3e-4 to 1.5e-3 per epoch, and each
# of those clears 1e-4, so `no_improve` never reaches `patience`. The cost bound is
# therefore the fixed `epochs` budget above, not early stopping.
#
# 25 epochs comes from the measured curve (single/pretrained/seed42, 26 epochs observed):
#   ep 2-5    2.74e-2 per epoch   <- nearly all of the gain, val 0.780 -> 0.671
#   ep 6-10   1.72e-3 per epoch
#   ep 16-20  1.14e-3 per epoch
#   ep 21-26  7.67e-4 per epoch   <- effectively flat
# Linear extrapolation from the last five epochs gives val 0.6307 at epoch 40, i.e. only
# 1.7% below epoch 26 for 15 more hours per run. 25 epochs sits past the point where the
# curve flattens, so the fixed budget costs ~1% of validation loss against a 40-epoch cap
# while cutting each run from 44 h to ~27.5 h.
early_stop_min_delta = 1e-4
resume_training = False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


is_ddp, rank, world_size, local_rank = init_ddp()
is_main = (not is_ddp) or rank == 0

if is_ddp:
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

if is_main:
    print(f"Device: {device}")
    print(f"DDP: {is_ddp} | rank={rank}/{world_size}")
    print(f"[Init] ARM={ARM} | cellfm_checkpoint={cellfm_checkpoint}")

set_seed(MODEL_SEED)


def ddp_reduce(total: float, steps: int):
    if not is_ddp:
        return total, steps
    t = torch.tensor([total, steps], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t[0].item()), int(t[1].item())


train_sampler = None
val_sampler = None
if is_ddp:
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    num_workers=num_workers,
    collate_fn=mil_collate,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(num_workers > 0),
)
val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    sampler=val_sampler,
    num_workers=num_workers,
    collate_fn=mil_collate,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(num_workers > 0),
)

sample_batch = next(iter(train_loader))
input_dim = sample_batch[KEYS.X].shape[-1]

model = MorphoCellfmMIL(
    input_dim=input_dim,
    n_genes=len(kept_genes),
    cellfm_dim=1536,
    cellfm_layers=2,
    cellfm_heads=48,
    cellfm_checkpoint=cellfm_checkpoint,  # None => random init
    freeze_cellfm=True,
    use_mock=False,
    use_retention=True,
    vocab_size=len(vocab),
    dropout=0.1,
    aggregation="mean",
    dispersion="gene",
).to(device)

# Identical trainable scope in both arms.
for p in model.cellfm.model.parameters():
    p.requires_grad = False
model.cellfm.model.gene_emb.requires_grad = True
for p in model.cellfm.model.cellwise_dec.map.parameters():
    p.requires_grad = True
model.gene_dispersion.requires_grad = True

if is_ddp:
    model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

base_model = model.module if is_ddp else model

optimizer = torch.optim.AdamW(
    [
        {"params": [base_model.cellfm.model.gene_emb], "lr": lr * gene_lr_mult},
        {"params": base_model.cellfm.model.cellwise_dec.map.parameters(), "lr": lr * gene_lr_mult},
        {"params": base_model.adapter.parameters(), "lr": lr},
        {"params": [base_model.gene_dispersion], "lr": lr},
    ],
    weight_decay=1e-2,
)

use_amp_flag = bool(use_amp and device.type == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp_flag)

gene_count = len(kept_genes)
gene_chunks = [
    list(range(i, min(i + gene_chunk_size, gene_count)))
    for i in range(0, gene_count, gene_chunk_size)
]

if is_main:
    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable}")
    print(f"Total genes: {gene_count} | chunks: {len(gene_chunks)} | chunk size: {gene_chunk_size}")
    print(f"Batches/epoch: {len(train_loader)} | chunk-steps/epoch: {len(train_loader) * len(gene_chunks)}")


# %%
# ----------------------------
# 3b) Optional throughput probe
# ----------------------------
if PROBE_STEPS > 0:
    # Warm-up so the measurement excludes cuDNN autotune / graph capture.
    warmup = min(5, PROBE_STEPS)
    it = iter(train_loader)
    for i in range(warmup + PROBE_STEPS):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        for k in batch:
            batch[k] = batch[k].to(device)
        ci = i % len(gene_chunks)
        idx = torch.as_tensor(gene_chunks[ci], device=device)
        batch_chunk = {
            KEYS.X: batch[KEYS.X],
            KEYS.PTR_BAG_INSTANCE: batch[KEYS.PTR_BAG_INSTANCE],
            KEYS.SIZE_FACTOR: batch[KEYS.SIZE_FACTOR],
            KEYS.GENE_IDS: batch[KEYS.GENE_IDS][idx],
            KEYS.Y_BAG: batch[KEYS.Y_BAG][:, idx],
        }
        if i == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp_flag):
            mu_bag, _ = model(batch_chunk)
            theta = torch.exp(base_model.gene_dispersion[idx])
            loss = nb_negative_log_likelihood(batch_chunk[KEYS.Y_BAG], mu_bag, theta)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    per_step = elapsed / PROBE_STEPS
    batches_per_epoch = len(train_loader)
    chunks = len(gene_chunks)
    steps_per_epoch = batches_per_epoch * chunks

    probe = {
        "exp_mode": EXP_MODE,
        "arm": ARM,
        "world_size": world_size,
        "batch_size": batch_size,
        "probe_steps": PROBE_STEPS,
        "elapsed_seconds": elapsed,
        "seconds_per_chunk_step": per_step,
        "steps_per_epoch": steps_per_epoch,
        "est_epoch_seconds": per_step * steps_per_epoch,
        "est_epoch_minutes": per_step * steps_per_epoch / 60.0,
        "train_spots": len(train_ds),
        "val_spots": len(val_ds),
        "effective_batch": batch_size * world_size,
    }
    if is_main:
        print("[PROBE] " + json.dumps(probe, indent=2))
        (CKPT_DIR / "probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
    if is_ddp:
        dist.destroy_process_group()
    sys.exit(0)


# %%
# ----------------------------
# 4) Training loop
# ----------------------------
best_val = 1e9
no_improve = 0
best_epoch = -1
epochs_run = 0
history = []

for epoch in range(1, epochs + 1):
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    model.train()
    total = 0.0
    steps = 0

    for batch_idx, batch in enumerate(train_loader):
        for k in batch:
            batch[k] = batch[k].to(device)

        chunk_order = torch.randperm(len(gene_chunks)).tolist()
        for ci in chunk_order:
            idx = torch.as_tensor(gene_chunks[ci], device=device)
            batch_chunk = {
                KEYS.X: batch[KEYS.X],
                KEYS.PTR_BAG_INSTANCE: batch[KEYS.PTR_BAG_INSTANCE],
                KEYS.SIZE_FACTOR: batch[KEYS.SIZE_FACTOR],
                KEYS.GENE_IDS: batch[KEYS.GENE_IDS][idx],
                KEYS.Y_BAG: batch[KEYS.Y_BAG][:, idx],
            }

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp_flag):
                mu_bag, _ = model(batch_chunk)
                theta = torch.exp(base_model.gene_dispersion[idx])
                loss = nb_negative_log_likelihood(batch_chunk[KEYS.Y_BAG], mu_bag, theta)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total += float(loss.item())
            steps += 1

        if is_main and (batch_idx % heartbeat_batches == 0):
            print(f"[{RUN_NAME}] epoch {epoch} batch {batch_idx}/{len(train_loader)}")

    total, steps = ddp_reduce(total, steps)
    train_loss = total / max(steps, 1)

    model.eval()
    total = 0.0
    steps = 0
    with torch.no_grad():
        for batch in val_loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            for ci in range(len(gene_chunks)):
                idx = torch.as_tensor(gene_chunks[ci], device=device)
                batch_chunk = {
                    KEYS.X: batch[KEYS.X],
                    KEYS.PTR_BAG_INSTANCE: batch[KEYS.PTR_BAG_INSTANCE],
                    KEYS.SIZE_FACTOR: batch[KEYS.SIZE_FACTOR],
                    KEYS.GENE_IDS: batch[KEYS.GENE_IDS][idx],
                    KEYS.Y_BAG: batch[KEYS.Y_BAG][:, idx],
                }
                with torch.cuda.amp.autocast(enabled=use_amp_flag):
                    mu_bag, _ = model(batch_chunk)
                    theta = torch.exp(base_model.gene_dispersion[idx])
                    loss = nb_negative_log_likelihood(batch_chunk[KEYS.Y_BAG], mu_bag, theta)
                total += float(loss.item())
                steps += 1

    total, steps = ddp_reduce(total, steps)
    val_loss = total / max(steps, 1)

    epochs_run = epoch
    history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    if is_main:
        print(f"[{RUN_NAME}] Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}")

    should_stop = False
    if is_main:
        if val_loss < best_val - early_stop_min_delta:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(base_model.state_dict(), CKPT_DIR / "best_model.pt")
            (CKPT_DIR / "common_genes.json").write_text(json.dumps(kept_genes, indent=2), encoding="utf-8")
            split_payload = {
                "exp_mode": EXP_MODE,
                "arm": ARM,
                "random_init": ARM == "random",
                "model_seed": MODEL_SEED,
                "split_seed": SPLIT_SEED,
                "train": train_ids,
                "val": val_ids,
                "test": [test_id],
                "within_slide_val_split": split_meta,
            }
            (CKPT_DIR / "split.json").write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
            print(f"[{RUN_NAME}] Saved best model (epoch {epoch}) ->", CKPT_DIR / "best_model.pt")
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                should_stop = True
                print(f"[{RUN_NAME}] Early stop: no improvement for {early_stop_patience} epochs")

    if is_ddp:
        flag = torch.tensor(1 if should_stop else 0, device=device)
        dist.broadcast(flag, src=0)
        should_stop = bool(flag.item())

    if should_stop:
        break

if is_main:
    summary = {
        "run_name": RUN_NAME,
        "exp_mode": EXP_MODE,
        "arm": ARM,
        "model_seed": MODEL_SEED,
        "split_seed": SPLIT_SEED,
        "batch_size": batch_size,
        "lr": lr,
        "epochs_cap": epochs,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta,
        "gene_chunk_size": gene_chunk_size,
        "world_size": world_size,
        "effective_batch": batch_size * world_size,
        "train_spots": len(train_ds),
        "val_spots": len(val_ds),
        "best_val": float(best_val),
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "hit_epoch_cap": bool(best_epoch >= epochs - early_stop_patience),
        "within_slide_val_split": split_meta,
    }
    (CKPT_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Full validation curve, so the plateau that justified stopping is on the record.
    with open(CKPT_DIR / "train_log.csv", "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss\n")
        for row in history:
            f.write(f"{row['epoch']},{row['train_loss']:.6f},{row['val_loss']:.6f}\n")
    print("Training done:", json.dumps(summary, indent=2))

if is_ddp:
    dist.destroy_process_group()
