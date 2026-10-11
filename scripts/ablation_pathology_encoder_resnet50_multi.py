from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import ablation_pathology_encoder_resnet50_single as single  # noqa: E402
import prostate_protocol_b_morpho as core  # noqa: E402


PROSTATE_SLIDES = ["INT25", "INT26", "INT27", "INT28"]
ENCODER_NAME = single.ENCODER_NAME
DEFAULT_TAG = "resnet50_pca576_multi_INT25_INT26_val_INT27_to_INT28"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pathology encoder ablation for the prostate multi-slide split: "
            "train INT25+INT26, validate INT27, test INT28, replacing HIPT "
            "features with ImageNet ResNet50 PCA features while keeping the "
            "CellFM backend fixed."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--mode", choices=["cache", "train", "predict", "summary", "all"], default="all")
    parser.add_argument("--train-slides", default="INT25,INT26")
    parser.add_argument("--val-slides", default="INT27")
    parser.add_argument("--test-slides", default="INT28")
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-predict", action="store_true")
    parser.add_argument("--keep-raw-resnet", action="store_true")
    parser.add_argument("--resnet-weights", type=Path, default=None)
    parser.add_argument("--pca-components", type=int, default=576)
    parser.add_argument("--pca-samples", type=int, default=100_000)
    parser.add_argument("--feature-tile-size", type=int, default=2048)
    parser.add_argument("--feature-batch-rows", type=int, default=16_384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--predict-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gene-chunk-size", type=int, default=896)
    parser.add_argument("--gene-lr-mult", type=float, default=3.0)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-dim", type=int, default=16000)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--heartbeat-batches", type=int, default=5)
    parser.add_argument("--cache-wait-timeout-min", type=float, default=720.0)
    return parser.parse_args()


def split_slide_list(arg: str, name: str) -> list[str]:
    slides = [x.strip() for x in arg.split(",") if x.strip()]
    if not slides:
        raise ValueError(f"{name} must not be empty")
    unknown = sorted(set(slides) - set(PROSTATE_SLIDES))
    if unknown:
        raise ValueError(f"Unknown prostate slide IDs in {name}: {unknown}")
    return slides


def unique_in_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def encoder_cache_root(root: Path, args: argparse.Namespace, train_slides: list[str]) -> Path:
    if args.cache_root is not None:
        return args.cache_root.expanduser().resolve()
    train_part = "_".join(train_slides)
    return (root / "data" / "superres_cache_encoder_ablation" / f"{ENCODER_NAME}_multi_{train_part}").resolve()


def projection_path(out_cache_root: Path, train_slides: list[str], n_components: int) -> Path:
    train_part = "_".join(train_slides)
    return out_cache_root / f"{ENCODER_NAME}_pca{n_components}_train_{train_part}.pkl"


def checkpoint_dir(root: Path, tag: str) -> Path:
    return root / "checkpoints" / "st_superres" / "pathology_encoder_ablation" / tag


def result_dir(root: Path, tag: str) -> Path:
    return root / "results" / "ablation_pathology_encoder" / tag


def best_model_path(root: Path, tag: str) -> Path:
    return checkpoint_dir(root, tag) / "best_model.pt"


def best_model_ready(root: Path, tag: str) -> bool:
    path = best_model_path(root, tag)
    return path.exists() and path.stat().st_size > 0


def require_best_model(root: Path, tag: str) -> Path:
    path = best_model_path(root, tag)
    if not path.exists():
        raise FileNotFoundError(f"Best model not found; skip predict/summary: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"Best model exists but is empty; skip predict/summary: {path}")
    return path


def fit_or_load_projection_multi(
    train_raw_paths: dict[str, Path],
    out_cache_root: Path,
    train_slides: list[str],
    n_components: int,
    n_samples: int,
    seed: int,
    force: bool,
):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    proj_path = projection_path(out_cache_root, train_slides, n_components)
    if proj_path.exists() and not force:
        with open(proj_path, "rb") as f:
            payload = pickle.load(f)
        print(f"[cache] loaded multi-slide projection: {proj_path}")
        return payload["scaler"], payload["pca"]

    rng = np.random.default_rng(seed)
    sample_parts = []
    remaining = int(n_samples)
    for idx, slide_id in enumerate(train_slides):
        raw = np.load(train_raw_paths[slide_id], mmap_mode="r")
        flat = raw.reshape(-1, raw.shape[-1])
        slots_left = len(train_slides) - idx
        take = min(flat.shape[0], max(1, math.ceil(remaining / max(slots_left, 1))))
        sample_idx = rng.choice(flat.shape[0], size=take, replace=False)
        sample_parts.append(np.asarray(flat[sample_idx], dtype=np.float32))
        remaining = max(0, remaining - take)
        print(f"[cache] sampled {take} grid locations from {slide_id}")

    sample = np.concatenate(sample_parts, axis=0)
    if sample.shape[0] < n_components:
        raise ValueError(f"PCA needs at least {n_components} samples, got {sample.shape[0]}")
    scaler = StandardScaler(copy=True)
    sample_z = scaler.fit_transform(sample)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    pca.fit(sample_z)
    out_cache_root.mkdir(parents=True, exist_ok=True)
    with open(proj_path, "wb") as f:
        pickle.dump(
            {
                "scaler": scaler,
                "pca": pca,
                "seed": seed,
                "sample_n": int(sample.shape[0]),
                "train_slides": train_slides,
                "raw_paths": {sid: str(path) for sid, path in train_raw_paths.items()},
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"[cache] fitted and saved multi-slide projection: {proj_path}")
    print(f"[cache] PCA explained variance sum: {float(np.sum(pca.explained_variance_ratio_)):.4f}")
    return scaler, pca


def projected_cache_ready(cache_root: Path, slides: list[str], train_slides: list[str], n_components: int) -> bool:
    if not projection_path(cache_root, train_slides, n_components).exists():
        return False
    return all((cache_root / sid / "embeddings-hist.pickle").exists() for sid in slides)


def build_resnet_cache_multi(
    args: argparse.Namespace,
    root: Path,
    train_slides: list[str],
    val_slides: list[str],
    test_slides: list[str],
) -> None:
    import torch

    slides = unique_in_order(train_slides + val_slides + test_slides)
    out_cache_root = encoder_cache_root(root, args, train_slides)
    if projected_cache_ready(out_cache_root, slides, train_slides, args.pca_components) and not args.force_cache:
        print(f"[cache] projected ResNet multi cache already complete: {out_cache_root}")
        return

    base_cache_root = root / "data" / "superres_cache"
    weights_path = args.resnet_weights or (root / "checkpoints" / "ResNet" / "resnet50-0676ba61.pth")
    if not weights_path.exists():
        raise FileNotFoundError(f"ResNet weights not found: {weights_path}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    out_cache_root.mkdir(parents=True, exist_ok=True)
    raw_paths = {}
    for sid in slides:
        raw_paths[sid] = single.extract_resnet_raw_features(
            sid,
            base_cache_root=base_cache_root,
            out_cache_root=out_cache_root,
            weights_path=weights_path,
            device=device,
            tile_size=args.feature_tile_size,
            force=args.force_cache,
        )

    scaler, pca = fit_or_load_projection_multi(
        {sid: raw_paths[sid] for sid in train_slides},
        out_cache_root=out_cache_root,
        train_slides=train_slides,
        n_components=args.pca_components,
        n_samples=args.pca_samples,
        seed=args.seed,
        force=args.force_cache,
    )
    for sid in slides:
        single.project_and_write_cache(
            sid,
            raw_paths[sid],
            base_cache_root=base_cache_root,
            out_cache_root=out_cache_root,
            scaler=scaler,
            pca=pca,
            batch_rows=args.feature_batch_rows,
            force=args.force_cache,
        )
        if not args.keep_raw_resnet:
            try:
                raw_paths[sid].unlink()
                print(f"[cache:{sid}] removed raw feature file: {raw_paths[sid]}")
            except FileNotFoundError:
                pass


def load_gene_setup_multi(
    root: Path,
    ckpt_dir: Path,
    train_slides: list[str],
    val_slides: list[str],
    test_slides: list[str],
    write_metadata: bool = True,
):
    from st_pipeline.data.gene_vocab import load_gene_vocab, map_genes_to_vocab
    from st_pipeline.superres.feature_cache import get_common_genes

    gene_vocab_path = root / "assets" / "cellfm" / "expand_gene_info.csv"
    gene_source_slides = unique_in_order(train_slides + val_slides)
    common_genes = get_common_genes(gene_source_slides, root=root)
    vocab = load_gene_vocab(gene_vocab_path)
    kept_genes, gene_ids = map_genes_to_vocab(common_genes, vocab)
    if write_metadata:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "common_genes.json").write_text(json.dumps(kept_genes, indent=2), encoding="utf-8")
        (ckpt_dir / "split_config.json").write_text(
            json.dumps(
                {
                    "protocol": "prostate_multi_split",
                    "encoder": ENCODER_NAME,
                    "train_slide_ids": train_slides,
                    "val_slide_ids": val_slides,
                    "test_slide_ids": test_slides,
                    "gene_source_slide_ids": gene_source_slides,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return kept_genes, gene_ids, vocab


def build_ablation_slide_dataset(
    slide_id: str,
    cache_root: Path,
    kept_genes: list[str],
    gene_ids,
):
    from st_pipeline.superres.feature_cache import normalize_slide

    emb_path = cache_root / slide_id / "embeddings-hist.pickle"
    if not emb_path.exists():
        raise FileNotFoundError(f"Ablation cache missing for {slide_id}: {emb_path}")
    embs, cnts, locs, radius, _, spot_ids = single.load_ablation_cache_slide(
        slide_id,
        root=cache_root.parent.parent.parent,
        cache_root=cache_root,
        genes=kept_genes,
    )
    embs_norm, _, stats = normalize_slide(embs, cnts)
    ds = core.SuperresSpotDataset(embs_norm, cnts, locs, radius, gene_ids)
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
        "encoder": ENCODER_NAME,
        "cache_root": str(cache_root),
    }
    return ds, meta


def train_multi(
    args: argparse.Namespace,
    root: Path,
    train_slides: list[str],
    val_slides: list[str],
    test_slides: list[str],
) -> bool:
    core.require_torch()
    import torch
    from st_pipeline.constants import KEYS
    from st_pipeline.data.collate import mil_collate
    from st_pipeline.model.nb_loss import nb_negative_log_likelihood
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import ConcatDataset, DataLoader
    from torch.utils.data.distributed import DistributedSampler

    ckpt_dir = checkpoint_dir(root, args.tag)
    cache_root = encoder_cache_root(root, args, train_slides)
    core.seed_everything(args.seed)
    is_ddp, rank, world_size, local_rank = core.init_ddp()
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() and is_ddp
        else "cuda:0" if torch.cuda.is_available()
        else "cpu"
    )
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if rank == 0:
        print(f"ROOT: {root}")
        print(f"Training pathology encoder multi ablation: {args.tag}")
        print(f"Train slides: {train_slides} | Val slides: {val_slides} | Test slides: {test_slides}")
        print(f"Device: {device} | DDP={is_ddp} rank={rank}/{world_size}")
        print(f"Cache root: {cache_root}")

    kept_genes, gene_ids, vocab = load_gene_setup_multi(
        root,
        ckpt_dir,
        train_slides,
        val_slides,
        test_slides,
        write_metadata=(rank == 0),
    )
    all_needed = unique_in_order(train_slides + val_slides)
    missing = [sid for sid in all_needed if not (cache_root / sid / "embeddings-hist.pickle").exists()]
    if missing:
        raise FileNotFoundError(f"Missing projected ablation caches for training/validation: {missing}")

    datasets = {}
    metas = {}
    for sid in all_needed:
        ds, meta = build_ablation_slide_dataset(sid, cache_root, kept_genes, gene_ids)
        datasets[sid] = ds
        metas[sid] = meta
        if rank == 0:
            np.savez(
                ckpt_dir / f"stats_{sid}.npz",
                radius_px=meta["radius_px"],
                mask_size=meta["mask_size"],
                embs_mean=meta["embs_mean"],
                embs_std=meta["embs_std"],
            )

    train_ds = ConcatDataset([datasets[sid] for sid in train_slides])
    val_ds = ConcatDataset([datasets[sid] for sid in val_slides])

    if rank == 0:
        split_payload = {
            "protocol": "prostate_multi_split",
            "encoder": ENCODER_NAME,
            "train_slide_ids": train_slides,
            "val_slide_ids": val_slides,
            "test_slide_ids": test_slides,
            "train_spots_per_slide": {sid: len(datasets[sid]) for sid in train_slides},
            "val_spots_per_slide": {sid: len(datasets[sid]) for sid in val_slides},
            "cache_root": str(cache_root),
        }
        (ckpt_dir / "split.json").write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
        print(f"Train spots: {len(train_ds)} | Val spots: {len(val_ds)}")

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
    model = core.build_model(input_dim, len(kept_genes), len(vocab), root, device)
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
    chunks = core.gene_chunks(len(kept_genes), args.gene_chunk_size)
    best_val = 1e9
    no_improve = 0

    if rank == 0:
        trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable}")
        print(f"Genes: {len(kept_genes)} | chunks: {len(chunks)} | gene_chunk_size={args.gene_chunk_size}")

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

        total, steps = core.ddp_reduce(is_ddp, device, total, steps)
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
        total, steps = core.ddp_reduce(is_ddp, device, total, steps)
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
            core.dist.broadcast(flag, src=0)
            should_stop = bool(flag.item())
        if should_stop:
            break

    should_continue = rank == 0 and best_model_ready(root, args.tag)
    if is_ddp:
        core.dist.destroy_process_group()
    return should_continue


def predict_multi(
    args: argparse.Namespace,
    root: Path,
    train_slides: list[str],
    val_slides: list[str],
    test_slides: list[str],
) -> None:
    core.require_torch()
    import torch
    from st_pipeline.constants import KEYS
    from st_pipeline.data.collate import mil_collate
    from st_pipeline.data.gene_vocab import load_gene_vocab, map_genes_to_vocab
    from torch.utils.data import DataLoader

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    ckpt_path = require_best_model(root, args.tag)
    core.seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ckpt_dir = checkpoint_dir(root, args.tag)
    cache_root = encoder_cache_root(root, args, train_slides)
    out_root = result_dir(root, args.tag)
    out_root.mkdir(parents=True, exist_ok=True)

    common_genes = json.loads((ckpt_dir / "common_genes.json").read_text(encoding="utf-8"))
    vocab = load_gene_vocab(root / "assets" / "cellfm" / "expand_gene_info.csv")
    kept_genes, gene_ids = map_genes_to_vocab(common_genes, vocab)
    chunks = core.gene_chunks(len(kept_genes), args.gene_chunk_size)

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
            metrics = core.save_metrics(pred, true, kept_genes, slide_out)
            rows.append({"test_slide_id": sid, **metrics})
            continue

        t0 = time.perf_counter()
        ds, meta = build_ablation_slide_dataset(sid, cache_root, kept_genes, gene_ids)
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
            model = core.build_model(input_dim, len(kept_genes), len(vocab), root, device)
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
                    chunk_preds.append(mu_bag.float().cpu().numpy())
                pred[offset : offset + batch_size] = np.concatenate(chunk_preds, axis=1)
                offset += batch_size
                if tqdm is None and b_idx % args.heartbeat_batches == 0:
                    print(f"[{sid}] predict batch {b_idx}/{len(loader)}", flush=True)

        slide_out.mkdir(parents=True, exist_ok=True)
        np.save(pred_path, pred)
        np.save(true_path, true)
        spot_ids = np.asarray(meta["spot_ids"], dtype=str)[ds.keep_idx]
        (slide_out / "spot_ids.txt").write_text("\n".join(map(str, spot_ids)), encoding="utf-8")
        metrics = core.save_metrics(pred, true, kept_genes, slide_out)
        rows.append({"test_slide_id": sid, **metrics})
        print(f"[{sid}] done in {time.perf_counter() - t0:.1f}s | mean={metrics['Mean_Pearson']:.4f}")

    core.write_fold_summary(rows, out_root)


def aggregate_multi(root: Path, tag: str, train_slides: list[str], val_slides: list[str], test_slides: list[str]) -> None:
    out_root = result_dir(root, tag)
    rows = []
    for sid in test_slides:
        metrics_path = out_root / sid / "metrics.json"
        if not metrics_path.exists():
            rows.append(
                {
                    "method": "Morpho-FM",
                    "encoder": "ImageNet_ResNet50_PCA576",
                    "protocol": "prostate_multi_split",
                    "train_slides": ",".join(train_slides),
                    "val_slides": ",".join(val_slides),
                    "test_slide_id": sid,
                    "status": "missing_metrics",
                    "result_dir": str(out_root / sid),
                }
            )
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "method": "Morpho-FM",
                "encoder": "ImageNet_ResNet50_PCA576",
                "protocol": "prostate_multi_split",
                "train_slides": ",".join(train_slides),
                "val_slides": ",".join(val_slides),
                "test_slide_id": sid,
                "status": "complete",
                "result_dir": str(out_root / sid),
                **metrics,
            }
        )
    df = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / "fold_status.csv", index=False)
    df.to_csv(out_root / "pathology_encoder_multi_summary.csv", index=False)
    print(f"Saved fold status: {out_root / 'fold_status.csv'}")


def summarize(args: argparse.Namespace, root: Path, train_slides: list[str], val_slides: list[str], test_slides: list[str]) -> None:
    require_best_model(root, args.tag)
    rows = []
    hipt_dir = root / "results" / "st_superres" / "cellfm_INT28"
    if (hipt_dir / "pred_bag.npy").exists() and (hipt_dir / "true_bag.npy").exists():
        pred = np.load(hipt_dir / "pred_bag.npy")
        true = np.load(hipt_dir / "true_bag.npy")
        corr = single.safe_pearson_vec(true, pred)
        rows.append(
            {
                "encoder": "HIPT_pathology_FM",
                "protocol": "prostate_multi_split",
                "train_slides": ",".join(train_slides),
                "val_slides": ",".join(val_slides),
                "test_slide_id": "INT28",
                "n_spots": int(pred.shape[0]),
                "n_genes": int(pred.shape[1]),
                "mean_pearson": float(np.nanmean(corr)),
                "median_pearson": float(np.nanmedian(corr)),
                "result_dir": str(hipt_dir),
                "status": "complete",
            }
        )
    else:
        rows.append(
            {
                "encoder": "HIPT_pathology_FM",
                "protocol": "prostate_multi_split",
                "train_slides": ",".join(train_slides),
                "val_slides": ",".join(val_slides),
                "test_slide_id": "INT28",
                "result_dir": str(hipt_dir),
                "status": "missing_baseline",
            }
        )

    out_root = result_dir(root, args.tag)
    for sid in test_slides:
        slide_dir = out_root / sid
        pred_path = slide_dir / "pred_bag.npy"
        true_path = slide_dir / "true_bag.npy"
        if not pred_path.exists() or not true_path.exists():
            rows.append(
                {
                    "encoder": "ImageNet_ResNet50_PCA576",
                    "protocol": "prostate_multi_split",
                    "train_slides": ",".join(train_slides),
                    "val_slides": ",".join(val_slides),
                    "test_slide_id": sid,
                    "status": "missing_prediction",
                    "result_dir": str(slide_dir),
                }
            )
            continue
        pred = np.load(pred_path)
        true = np.load(true_path)
        corr = single.safe_pearson_vec(true, pred)
        rows.append(
            {
                "encoder": "ImageNet_ResNet50_PCA576",
                "protocol": "prostate_multi_split",
                "train_slides": ",".join(train_slides),
                "val_slides": ",".join(val_slides),
                "test_slide_id": sid,
                "n_spots": int(pred.shape[0]),
                "n_genes": int(pred.shape[1]),
                "mean_pearson": float(np.nanmean(corr)),
                "median_pearson": float(np.nanmedian(corr)),
                "result_dir": str(slide_dir),
                "status": "complete",
            }
        )

    df = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    out = out_root / "pathology_encoder_ablation_summary.csv"
    df.to_csv(out, index=False)
    print(f"Saved pathology encoder summary: {out}")
    print(df)


def cache_sentinel(cache_root: Path, tag: str) -> Path:
    return cache_root / f".cache_complete_{tag}.json"


def wait_for_cache_ready(path: Path, timeout_min: float) -> None:
    deadline = time.time() + timeout_min * 60.0
    while time.time() < deadline:
        if path.exists():
            print(f"[cache] detected cache sentinel: {path}")
            return
        time.sleep(30)
    raise TimeoutError(f"Timed out waiting for cache sentinel: {path}")


def maybe_build_cache_with_torchrun_guard(
    args: argparse.Namespace,
    root: Path,
    train_slides: list[str],
    val_slides: list[str],
    test_slides: list[str],
) -> None:
    cache_root = encoder_cache_root(root, args, train_slides)
    sentinel = cache_sentinel(cache_root, args.tag)
    rank_env = int(os.environ.get("RANK", "0"))
    if rank_env == 0:
        build_resnet_cache_multi(args, root, train_slides, val_slides, test_slides)
        cache_root.mkdir(parents=True, exist_ok=True)
        sentinel.write_text(
            json.dumps(
                {
                    "tag": args.tag,
                    "encoder": ENCODER_NAME,
                    "train_slides": train_slides,
                    "val_slides": val_slides,
                    "test_slides": test_slides,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        wait_for_cache_ready(sentinel, args.cache_wait_timeout_min)


def main() -> None:
    args = parse_args()
    root = single.setup_root(args.root)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_slides = split_slide_list(args.train_slides, "--train-slides")
    val_slides = split_slide_list(args.val_slides, "--val-slides")
    test_slides = split_slide_list(args.test_slides, "--test-slides")
    overlap = (set(train_slides) & set(val_slides)) | (set(train_slides) & set(test_slides)) | (set(val_slides) & set(test_slides))
    if overlap:
        raise ValueError(f"Train/val/test slide sets must be disjoint, overlap={sorted(overlap)}")

    if args.mode in {"cache", "all"}:
        maybe_build_cache_with_torchrun_guard(args, root, train_slides, val_slides, test_slides)

    continue_after_train = True
    if args.mode in {"train", "all"}:
        continue_after_train = train_multi(args, root, train_slides, val_slides, test_slides)

    if args.mode in {"predict", "all"}:
        if continue_after_train and best_model_ready(root, args.tag):
            predict_multi(args, root, train_slides, val_slides, test_slides)
            aggregate_multi(root, args.tag, train_slides, val_slides, test_slides)
        else:
            print(f"[guard] Skip prediction because best model is not ready for tag={args.tag}")

    if args.mode in {"summary", "all"}:
        if continue_after_train and best_model_ready(root, args.tag):
            summarize(args, root, train_slides, val_slides, test_slides)
        else:
            print(f"[guard] Skip summary because best model is not ready for tag={args.tag}")


if __name__ == "__main__":
    main()
