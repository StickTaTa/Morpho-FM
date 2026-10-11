from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, Dataset, Subset
    from torch.utils.data.distributed import DistributedSampler
except (ModuleNotFoundError, ImportError):  # Allows --help and aggregation checks outside the training env.
    torch = None
    dist = None
    DDP = None
    DataLoader = None
    Dataset = object
    Subset = None
    DistributedSampler = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional progress bar
    tqdm = None


PROSTATE_SLIDES = ["INT25", "INT26", "INT27", "INT28"]
GRID_FACTOR = 16


def require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("torch is required for train/predict modes. Run this script in the Morpho-FM env.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Morpho-FM prostate protocol B: one training slide with blocked "
            "internal validation, then predict the other prostate slides."
        )
    )
    parser.add_argument("--root", type=Path, default=None, help="Project root. Defaults to MORPHO_VC_ROOT or cwd.")
    parser.add_argument("--train-slide", required=True, choices=PROSTATE_SLIDES)
    parser.add_argument(
        "--test-slides",
        default=None,
        help="Comma-separated test slide IDs. Defaults to all prostate slides except --train-slide.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Override checkpoint/result fold tag. Useful when predicting a subset of test slides from one full fold.",
    )
    parser.add_argument("--mode", choices=["train", "predict", "all", "aggregate"], default="all")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-predict", action="store_true")
    parser.add_argument("--resume-training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--predict-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gene-chunk-size", type=int, default=256)
    parser.add_argument("--gene-lr-mult", type=float, default=3.0)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-dim", type=int, default=16000)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--heartbeat-batches", type=int, default=5)
    return parser.parse_args()


def setup_root(root_arg: Path | None) -> Path:
    root = root_arg or Path(os.environ.get("MORPHO_VC_ROOT", "."))
    root = root.expanduser().resolve()
    os.chdir(root)
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.append(str(src))
    return root


def split_test_slides(train_slide: str, test_slides_arg: str | None) -> list[str]:
    if test_slides_arg:
        slides = [x.strip() for x in test_slides_arg.split(",") if x.strip()]
    else:
        slides = [sid for sid in PROSTATE_SLIDES if sid != train_slide]
    unknown = sorted(set(slides) - set(PROSTATE_SLIDES))
    if unknown:
        raise ValueError(f"Unknown prostate slide IDs: {unknown}")
    if train_slide in slides:
        raise ValueError("--test-slides must not include --train-slide")
    if not slides:
        raise ValueError("No test slides selected.")
    return slides


def fold_tag(train_slide: str, test_slides: list[str]) -> str:
    return f"{train_slide}_to_{'_'.join(test_slides)}"


def checkpoint_dir(root: Path, tag: str) -> Path:
    return root / "checkpoints" / "st_superres" / "protocol_b_rotating" / f"cellfm_single_{tag}"


def result_dir(root: Path, tag: str) -> Path:
    return root / "results" / "prostate_hest" / "morpho" / "single_superres" / tag


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_ddp() -> tuple[bool, int, int, int]:
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


def get_disk_mask(radius: float) -> np.ndarray:
    radius_ceil = int(np.ceil(radius))
    locs = np.meshgrid(
        np.arange(-radius_ceil, radius_ceil + 1),
        np.arange(-radius_ceil, radius_ceil + 1),
        indexing="ij",
    )
    locs = np.stack(locs, axis=-1)
    return (locs**2).sum(-1) <= radius**2


class SuperresSpotDataset(Dataset):
    def __init__(self, embs, cnts, locs_px, radius_px, gene_ids, grid_factor: int = GRID_FACTOR):
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

        h, w = embs.shape[:2]
        r0, r1 = self._patch_offsets()
        valid = (
            (locs_grid[:, 0] + r0[0] >= 0)
            & (locs_grid[:, 0] + r0[1] <= h)
            & (locs_grid[:, 1] + r1[0] >= 0)
            & (locs_grid[:, 1] + r1[1] <= w)
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

    def __len__(self) -> int:
        return len(self.locs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | float]:
        from st_pipeline.constants import KEYS

        i, j = self.locs[idx]
        r0, r1 = self._patch_offsets()
        patch = self.embs[i + r0[0] : i + r0[1], j + r1[0] : j + r1[1]]
        x = patch if self.mask.all() else patch[self.mask]
        return {
            KEYS.X: torch.from_numpy(x.astype(np.float32)),
            KEYS.Y_BAG: torch.from_numpy(self.cnts[idx].astype(np.float32)),
            KEYS.SIZE_FACTOR: float(self.size_factors[idx]),
            KEYS.GENE_IDS: self.gene_ids,
        }


def compute_blocks(coords: np.ndarray, grid_size: int) -> dict[tuple[int, int], list[int]]:
    min_x, min_y = coords.min(0)
    max_x, max_y = coords.max(0)
    spans = np.array([max_x - min_x + 1e-6, max_y - min_y + 1e-6])
    step = spans / grid_size
    blocks: dict[tuple[int, int], list[int]] = {}
    for i, (x, y) in enumerate(coords):
        gx = int((x - min_x) / step[0])
        gy = int((y - min_y) / step[1])
        gx = min(max(gx, 0), grid_size - 1)
        gy = min(max(gy, 0), grid_size - 1)
        blocks.setdefault((gx, gy), []).append(i)
    return blocks


def select_grid_size(coords: np.ndarray, min_grid=4, max_grid=10, min_blocks=20, max_blocks=80):
    target = (min_blocks + max_blocks) / 2.0
    candidates = []
    for grid_size in range(min_grid, max_grid + 1):
        blocks = compute_blocks(coords, grid_size)
        n_blocks = len(blocks)
        in_range = min_blocks <= n_blocks <= max_blocks
        score = 0 if in_range else abs(n_blocks - target)
        candidates.append((in_range, score, grid_size, blocks, n_blocks))
    candidates.sort(key=lambda x: (not x[0], x[1]))
    _, _, grid_size, blocks, n_blocks = candidates[0]
    return grid_size, blocks, n_blocks


def make_blocked_split(coords, val_ratio=0.2, seed=42, grid_size=None):
    coords = np.asarray(coords, dtype=np.float32)
    n_spots = coords.shape[0]
    if grid_size is None:
        grid_size, blocks, block_count = select_grid_size(coords)
    else:
        blocks = compute_blocks(coords, grid_size)
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
        remaining = [bid for bid in blocks if bid not in val_blocks]
        remaining.sort(key=lambda b: block_sizes[b], reverse=True)
        for bid in remaining:
            if len(val_blocks) >= len(blocks) - 1:
                break
            val_blocks.add(bid)
            val_count += block_sizes[bid]
            if val_count >= target_total:
                break

    val_indices = sorted({i for bid in val_blocks for i in blocks[bid]})
    val_set = set(val_indices)
    train_indices = [i for i in range(n_spots) if i not in val_set]
    return train_indices, val_indices, {
        "strategy": "blocked_grid_quadrant",
        "grid_size": int(grid_size),
        "block_count": int(block_count),
        "val_ratio_target": float(val_ratio),
        "val_ratio_actual": float(len(val_indices) / max(n_spots, 1)),
        "val_blocks": [f"{bid[0]}_{bid[1]}" for bid in sorted(val_blocks)],
    }


def load_gene_setup(root: Path, ckpt_dir: Path, write_metadata: bool = True):
    from st_pipeline.data.gene_vocab import load_gene_vocab, map_genes_to_vocab
    from st_pipeline.superres.feature_cache import get_common_genes

    gene_vocab_path = root / "assets" / "cellfm" / "expand_gene_info.csv"
    common_genes = get_common_genes(PROSTATE_SLIDES, root=root)
    vocab = load_gene_vocab(gene_vocab_path)
    kept_genes, gene_ids = map_genes_to_vocab(common_genes, vocab)
    if write_metadata:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "common_genes.json").write_text(json.dumps(kept_genes, indent=2), encoding="utf-8")
    return kept_genes, gene_ids, vocab


def build_slide_dataset(
    root: Path,
    slide_id: str,
    kept_genes,
    gene_ids,
    device,
    force_cache: bool,
    max_dim: int,
    ensure_cache: bool = True,
):
    from st_pipeline.superres.feature_cache import ensure_feature_cache, load_cache_slide, normalize_slide

    if ensure_cache:
        ensure_feature_cache(
            [slide_id],
            root=root,
            force=force_cache,
            device=device,
            max_dim=max_dim,
            use_tissue_seg=True,
        )
    embs, cnts, locs, radius, _, spot_ids = load_cache_slide(slide_id, root=root, genes=kept_genes)
    embs_norm, _, stats = normalize_slide(embs, cnts)
    ds = SuperresSpotDataset(embs_norm, cnts, locs, radius, gene_ids)
    meta = {
        "slide_id": slide_id,
        "embs_shape": list(embs.shape),
        "cnts_shape": list(cnts.shape),
        "locs_shape": list(locs.shape),
        "radius_px": float(radius),
        "mask_size": int(ds.mask_size),
        "embs_mean": stats["embs_mean"],
        "embs_std": stats["embs_std"],
        "spot_ids": spot_ids,
    }
    return ds, meta


def build_model(input_dim: int, n_genes: int, vocab_size: int, root: Path, device: torch.device):
    from st_pipeline.model.morpho_cellfm_mil import MorphoCellfmMIL

    model = MorphoCellfmMIL(
        input_dim=input_dim,
        n_genes=n_genes,
        cellfm_dim=1536,
        cellfm_layers=2,
        cellfm_heads=48,
        cellfm_checkpoint=str(root / "checkpoints" / "CellFM" / "CellFM_80M_weight.pt"),
        freeze_cellfm=True,
        use_mock=False,
        use_retention=True,
        vocab_size=vocab_size,
        dropout=0.1,
        aggregation="mean",
        dispersion="gene",
    ).to(device)
    for p in model.cellfm.model.parameters():
        p.requires_grad = False
    model.cellfm.model.gene_emb.requires_grad = True
    for p in model.cellfm.model.cellwise_dec.map.parameters():
        p.requires_grad = True
    model.gene_dispersion.requires_grad = True
    return model


def gene_chunks(n_genes: int, chunk_size: int) -> list[list[int]]:
    if chunk_size <= 0:
        raise ValueError("--gene-chunk-size must be > 0")
    return [list(range(i, min(i + chunk_size, n_genes))) for i in range(0, n_genes, chunk_size)]


def ddp_reduce(is_ddp: bool, device: torch.device, total: float, steps: int) -> tuple[float, int]:
    if not is_ddp:
        return total, steps
    t = torch.tensor([total, steps], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t[0].item()), int(t[1].item())


def train_fold(args, root: Path, train_slide: str, test_slides: list[str], tag: str) -> bool:
    require_torch()
    from st_pipeline.constants import KEYS
    from st_pipeline.data.collate import mil_collate
    from st_pipeline.model.nb_loss import nb_negative_log_likelihood

    ckpt_dir = checkpoint_dir(root, tag)
    seed_everything(args.seed)
    is_ddp, rank, world_size, local_rank = init_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and is_ddp else "cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if rank == 0:
        print(f"ROOT: {root}")
        print(f"Training fold {tag}")
        print(f"Device: {device} | DDP={is_ddp} rank={rank}/{world_size}")

    kept_genes, gene_ids, vocab = load_gene_setup(root, ckpt_dir, write_metadata=(rank == 0))
    if rank == 0:
        build_slide_dataset(root, train_slide, kept_genes, gene_ids, device, args.force_cache, args.max_dim, ensure_cache=True)
    if is_ddp:
        dist.barrier()
    train_base, train_meta = build_slide_dataset(
        root,
        train_slide,
        kept_genes,
        gene_ids,
        device,
        force_cache=False,
        max_dim=args.max_dim,
        ensure_cache=False,
    )
    train_indices, val_indices, split_meta = make_blocked_split(
        train_base.locs_px,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    train_ds = Subset(train_base, train_indices)
    val_ds = Subset(train_base, val_indices)

    if rank == 0:
        np.savez(
            ckpt_dir / f"stats_{train_slide}.npz",
            radius_px=train_meta["radius_px"],
            mask_size=train_meta["mask_size"],
            embs_mean=train_meta["embs_mean"],
            embs_std=train_meta["embs_std"],
        )
        split_payload = {
            "protocol": "prostate_protocol_b_rotating_single_slide",
            "train_slide_id": train_slide,
            "test_slide_ids": test_slides,
            "slide_ids_all": PROSTATE_SLIDES,
            "validation_split": split_meta,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "train_spot_ids": [train_meta["spot_ids"][i] for i in train_base.keep_idx[train_indices]],
            "val_spot_ids": [train_meta["spot_ids"][i] for i in train_base.keep_idx[val_indices]],
        }
        (ckpt_dir / "split.json").write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
        print(f"Train spots: {len(train_ds)} | Val spots: {len(val_ds)}")
        print(f"Validation split: {split_meta}")

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=mil_collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=mil_collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[KEYS.X].shape[-1]
    model = build_model(input_dim, len(kept_genes), len(vocab), root, device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    base_model = model.module if is_ddp else model

    resume_path = ckpt_dir / "best_model.pt"
    if args.resume_training and resume_path.exists():
        base_model.load_state_dict(torch.load(resume_path, map_location=device))
        if rank == 0:
            print(f"Resumed from {resume_path}")

    optimizer = torch.optim.AdamW(
        [
            {"params": [base_model.cellfm.model.gene_emb], "lr": args.lr * args.gene_lr_mult},
            {"params": base_model.cellfm.model.cellwise_dec.map.parameters(), "lr": args.lr * args.gene_lr_mult},
            {"params": base_model.adapter.parameters(), "lr": args.lr},
            {"params": [base_model.gene_dispersion], "lr": args.lr},
        ],
        weight_decay=1e-2,
    )
    use_amp = bool(args.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    chunks = gene_chunks(len(kept_genes), args.gene_chunk_size)
    best_val = 1e9
    no_improve = 0

    if rank == 0:
        trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable}")
        print(f"Genes: {len(kept_genes)} | chunks: {len(chunks)}")

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        total = 0.0
        steps = 0
        for batch_idx, batch in enumerate(train_loader):
            for key in batch:
                batch[key] = batch[key].to(device, non_blocking=True)
            chunk_order = torch.randperm(len(chunks)).tolist()
            for ci in chunk_order:
                idx = torch.as_tensor(chunks[ci], device=device)
                batch_chunk = {
                    KEYS.X: batch[KEYS.X],
                    KEYS.PTR_BAG_INSTANCE: batch[KEYS.PTR_BAG_INSTANCE],
                    KEYS.SIZE_FACTOR: batch[KEYS.SIZE_FACTOR],
                    KEYS.GENE_IDS: batch[KEYS.GENE_IDS][idx],
                    KEYS.Y_BAG: batch[KEYS.Y_BAG][:, idx],
                }
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    mu_bag, _ = model(batch_chunk)
                    theta = torch.exp(base_model.gene_dispersion[idx])
                    loss = nb_negative_log_likelihood(batch_chunk[KEYS.Y_BAG], mu_bag, theta)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total += float(loss.item())
                steps += 1
            if rank == 0 and batch_idx % args.heartbeat_batches == 0:
                print(f"epoch {epoch} batch {batch_idx}/{len(train_loader)}", flush=True)

        total, steps = ddp_reduce(is_ddp, device, total, steps)
        train_loss = total / max(steps, 1)

        model.eval()
        total = 0.0
        steps = 0
        with torch.inference_mode():
            for batch in val_loader:
                for key in batch:
                    batch[key] = batch[key].to(device, non_blocking=True)
                for chunk in chunks:
                    idx = torch.as_tensor(chunk, device=device)
                    batch_chunk = {
                        KEYS.X: batch[KEYS.X],
                        KEYS.PTR_BAG_INSTANCE: batch[KEYS.PTR_BAG_INSTANCE],
                        KEYS.SIZE_FACTOR: batch[KEYS.SIZE_FACTOR],
                        KEYS.GENE_IDS: batch[KEYS.GENE_IDS][idx],
                        KEYS.Y_BAG: batch[KEYS.Y_BAG][:, idx],
                    }
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        mu_bag, _ = model(batch_chunk)
                        theta = torch.exp(base_model.gene_dispersion[idx])
                        loss = nb_negative_log_likelihood(batch_chunk[KEYS.Y_BAG], mu_bag, theta)
                    total += float(loss.item())
                    steps += 1
        total, steps = ddp_reduce(is_ddp, device, total, steps)
        val_loss = total / max(steps, 1)

        if rank == 0:
            print(f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}", flush=True)

        should_stop = False
        if rank == 0:
            if val_loss < best_val - args.early_stop_min_delta:
                best_val = val_loss
                no_improve = 0
                torch.save(base_model.state_dict(), ckpt_dir / "best_model.pt")
                (ckpt_dir / "best_metrics.json").write_text(
                    json.dumps({"best_val_loss": best_val, "epoch": epoch}, indent=2),
                    encoding="utf-8",
                )
                print(f"Saved best model: {ckpt_dir / 'best_model.pt'}")
            else:
                no_improve += 1
                if no_improve >= args.early_stop_patience:
                    should_stop = True
                    print(f"Early stop: no improvement for {no_improve} epochs")

        if is_ddp:
            flag = torch.tensor(1 if should_stop else 0, device=device)
            dist.broadcast(flag, src=0)
            should_stop = bool(flag.item())
        if should_stop:
            break

    should_continue = rank == 0
    if is_ddp:
        dist.destroy_process_group()
    return should_continue


def safe_pearson_vec(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    true = true.astype(np.float64, copy=False)
    pred = pred.astype(np.float64, copy=False)
    true_mean = true.mean(axis=0, keepdims=True)
    pred_mean = pred.mean(axis=0, keepdims=True)
    true_centered = true - true_mean
    pred_centered = pred - pred_mean
    num = (true_centered * pred_centered).sum(axis=0)
    den = np.sqrt((true_centered * true_centered).sum(axis=0) * (pred_centered * pred_centered).sum(axis=0))
    out = np.full(true.shape[1], np.nan, dtype=np.float32)
    valid = den > 0
    out[valid] = (num[valid] / den[valid]).astype(np.float32)
    return out


def save_metrics(pred: np.ndarray, true: np.ndarray, kept_genes: list[str], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if pred.shape != true.shape:
        n = min(pred.shape[0], true.shape[0])
        pred = pred[:n]
        true = true[:n]
    corr_all = safe_pearson_vec(true, pred)
    var_true = np.var(true, axis=0)
    var_pred = np.var(pred, axis=0)
    top_hvg = min(1000, true.shape[1])
    hvg_idx_true = np.argsort(var_true)[::-1][:top_hvg]
    hvg_idx_pred = np.argsort(var_pred)[::-1][:top_hvg]
    overlap_idx = np.array(sorted(set(hvg_idx_true) & set(hvg_idx_pred)), dtype=np.int64)
    corr_hvg = corr_all[hvg_idx_true]
    corr_overlap = corr_all[overlap_idx] if overlap_idx.size else np.array([], dtype=np.float32)
    valid = np.where(np.isfinite(corr_all))[0]
    best_idx = int(valid[np.nanargmax(corr_all[valid])]) if valid.size else None
    metrics = {
        "n_spots": int(pred.shape[0]),
        "n_genes": int(pred.shape[1]),
        "MAE": float(np.mean(np.abs(pred - true))),
        "RMSE": float(np.sqrt(np.mean((pred - true) ** 2))),
        "Mean_Pearson": float(np.nanmean(corr_all)) if valid.size else float("nan"),
        "Median_Pearson": float(np.nanmedian(corr_all)) if valid.size else float("nan"),
        "HVG1000_Mean_Pearson": float(np.nanmean(corr_hvg)),
        "HVG1000_Median_Pearson": float(np.nanmedian(corr_hvg)),
        "HVG1000_Overlap": int(overlap_idx.size),
        "Best_Gene": kept_genes[best_idx] if best_idx is not None else "NA",
        "Best_Pearson": float(corr_all[best_idx]) if best_idx is not None else None,
    }
    np.save(out_dir / "gene_corr_all.npy", corr_all)
    np.save(out_dir / "gene_corr_hvg.npy", corr_hvg)
    np.save(out_dir / "hvg_idx_true.npy", hvg_idx_true)
    np.save(out_dir / "hvg_idx_pred.npy", hvg_idx_pred)
    np.save(out_dir / "corr_overlap_genes.npy", corr_overlap)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def predict_fold(args, root: Path, train_slide: str, test_slides: list[str], tag: str) -> None:
    require_torch()
    from st_pipeline.constants import KEYS
    from st_pipeline.data.collate import mil_collate
    from st_pipeline.data.gene_vocab import load_gene_vocab, map_genes_to_vocab

    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    ckpt_dir = checkpoint_dir(root, tag)
    out_root = result_dir(root, tag)
    out_root.mkdir(parents=True, exist_ok=True)

    kept_genes = json.loads((ckpt_dir / "common_genes.json").read_text(encoding="utf-8"))
    vocab = load_gene_vocab(root / "assets" / "cellfm" / "expand_gene_info.csv")
    kept_genes, gene_ids = map_genes_to_vocab(kept_genes, vocab)
    chunks = gene_chunks(len(kept_genes), args.gene_chunk_size)

    rows = []
    model = None
    input_dim = None
    for sid in test_slides:
        slide_out = out_root / sid
        pred_path = slide_out / "pred_bag.npy"
        true_path = slide_out / "true_bag.npy"
        if pred_path.exists() and true_path.exists() and not args.force_predict:
            print(f"[{sid}] cached predictions found.")
            pred = np.load(pred_path)
            true = np.load(true_path)
            metrics = save_metrics(pred, true, kept_genes, slide_out)
            rows.append({"test_slide_id": sid, **metrics})
            continue

        t0 = time.perf_counter()
        ds, meta = build_slide_dataset(root, sid, kept_genes, gene_ids, device, args.force_cache, args.max_dim)
        loader = DataLoader(
            ds,
            batch_size=args.predict_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=mil_collate,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )
        if input_dim is None:
            sample_batch = next(iter(loader))
            input_dim = sample_batch[KEYS.X].shape[-1]
            model = build_model(input_dim, len(kept_genes), len(vocab), root, device)
            ckpt_path = ckpt_dir / "best_model.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
        assert model is not None

        pred = np.zeros((len(ds), len(kept_genes)), dtype=np.float32)
        true = np.zeros((len(ds), len(kept_genes)), dtype=np.float32)
        offset = 0
        iterator = tqdm(loader, total=len(loader), desc=f"{sid} predict", ncols=100) if tqdm else loader
        with torch.inference_mode():
            for b_idx, batch in enumerate(iterator):
                for key in batch:
                    batch[key] = batch[key].to(device, non_blocking=True)
                batch_size = batch[KEYS.Y_BAG].shape[0]
                true[offset : offset + batch_size] = batch[KEYS.Y_BAG].cpu().numpy()
                chunk_preds = []
                for chunk in chunks:
                    idx = torch.as_tensor(chunk, device=device)
                    batch_chunk = {
                        KEYS.X: batch[KEYS.X],
                        KEYS.PTR_BAG_INSTANCE: batch[KEYS.PTR_BAG_INSTANCE],
                        KEYS.SIZE_FACTOR: batch[KEYS.SIZE_FACTOR],
                        KEYS.GENE_IDS: batch[KEYS.GENE_IDS][idx],
                    }
                    with torch.amp.autocast("cuda", enabled=bool(args.use_amp and device.type == "cuda")):
                        mu_bag, _ = model(batch_chunk)
                    chunk_preds.append(mu_bag.cpu().numpy())
                pred[offset : offset + batch_size] = np.concatenate(chunk_preds, axis=1)
                offset += batch_size
                if tqdm is None and b_idx % args.heartbeat_batches == 0:
                    print(f"[{sid}] predict batch {b_idx}/{len(loader)}", flush=True)

        slide_out.mkdir(parents=True, exist_ok=True)
        np.save(pred_path, pred)
        np.save(true_path, true)
        spot_ids = np.asarray(meta["spot_ids"], dtype=str)[ds.keep_idx]
        (slide_out / "spot_ids.txt").write_text("\n".join(map(str, spot_ids)), encoding="utf-8")
        metrics = save_metrics(pred, true, kept_genes, slide_out)
        rows.append({"test_slide_id": sid, **metrics})
        print(f"[{sid}] done in {time.perf_counter() - t0:.1f}s | mean={metrics['Mean_Pearson']:.4f}")

    write_fold_summary(rows, out_root)


def write_fold_summary(rows: list[dict], out_root: Path) -> None:
    if not rows:
        print("[WARN] No prediction rows to summarize.")
        return
    df = pd.DataFrame(rows)
    out = out_root / "multi_test_summary.csv"
    df.to_csv(out, index=False)
    print(f"Saved summary: {out}")
    print(df)


def aggregate_fold(root: Path, train_slide: str, test_slides: list[str], tag: str) -> None:
    out_root = result_dir(root, tag)
    rows = []
    for sid in test_slides:
        metrics_path = out_root / sid / "metrics.json"
        if not metrics_path.exists():
            rows.append({
                "method": "Morpho-FM",
                "protocol": "prostate_protocol_b_rotating_single_slide",
                "train_slide": train_slide,
                "test_slide_id": sid,
                "status": "missing_metrics",
                "result_dir": str(out_root / sid),
            })
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({
            "method": "Morpho-FM",
            "protocol": "prostate_protocol_b_rotating_single_slide",
            "train_slide": train_slide,
            "validation_type": "blocked_grid_quadrant",
            "test_slide_id": sid,
            "status": "complete",
            "result_dir": str(out_root / sid),
            **metrics,
        })
    summary_root = root / "results" / "prostate_hest" / "protocol_b_rotating"
    summary_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "fold_status.csv", index=False)
    master_path = summary_root / f"{tag}_morpho_summary.csv"
    df.to_csv(master_path, index=False)
    print(f"Saved fold status: {out_root / 'fold_status.csv'}")
    print(f"Saved fold master summary: {master_path}")


def main() -> None:
    args = parse_args()
    root = setup_root(args.root)
    train_slide = args.train_slide
    test_slides = split_test_slides(train_slide, args.test_slides)
    tag = args.tag or fold_tag(train_slide, test_slides)

    continue_after_train = True
    if args.mode in {"train", "all"}:
        continue_after_train = train_fold(args, root, train_slide, test_slides, tag)
    if not continue_after_train:
        return
    if args.mode in {"predict", "all"}:
        predict_fold(args, root, train_slide, test_slides, tag)
    if args.mode in {"aggregate", "all", "predict"}:
        aggregate_fold(root, train_slide, test_slides, tag)


if __name__ == "__main__":
    main()
