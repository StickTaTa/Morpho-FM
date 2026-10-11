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

import prostate_protocol_b_morpho as core


KIDNEY_SLIDES = ["INT13", "INT14", "INT15", "INT17", "INT18", "INT19", "INT21", "INT24"]
SINGLE_TRAIN_SLIDES = ["INT18", "INT21", "INT24"]
PROTOCOLS = ["multi_loso", "single_rotating"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Morpho-FM kidney rotating benchmark folds.")
    parser.add_argument("--root", type=Path, default=None, help="Project root. Defaults to MORPHO_VC_ROOT or cwd.")
    parser.add_argument("--protocol", choices=PROTOCOLS, required=True)
    parser.add_argument("--test-slide", choices=KIDNEY_SLIDES, default=None, help="Held-out test slide for multi_loso.")
    parser.add_argument("--val-slide", choices=KIDNEY_SLIDES, default=None, help="Validation slide override for multi_loso.")
    parser.add_argument("--train-slide", choices=KIDNEY_SLIDES, default=None, help="Training slide for single_rotating.")
    parser.add_argument("--test-slides", default=None, help="Comma-separated test slides for prediction subsets.")
    parser.add_argument("--tag", default=None, help="Override checkpoint/result fold tag.")
    parser.add_argument("--mode", choices=["cache", "bench", "train", "predict", "all", "aggregate"], default="all")
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
    parser.add_argument("--bench-batch-sizes", default="4,8,16")
    parser.add_argument("--bench-gene-chunk-sizes", default="256,512")
    parser.add_argument("--probe-batches", type=int, default=3)
    return parser.parse_args()


def setup_root(root_arg: Path | None) -> Path:
    root = root_arg or Path(os.environ.get("MORPHO_VC_ROOT", "."))
    root = root.expanduser().resolve()
    os.chdir(root)
    src = root / "src"
    scripts = root / "scripts"
    for path in (src, scripts):
        if str(path) not in sys.path:
            sys.path.append(str(path))
    return root


def cyclic_val_slide(test_slide: str) -> str:
    idx = KIDNEY_SLIDES.index(test_slide)
    return KIDNEY_SLIDES[(idx + 1) % len(KIDNEY_SLIDES)]


def split_csv_slides(value: str | None) -> list[str] | None:
    if not value:
        return None
    slides = [x.strip() for x in value.split(",") if x.strip()]
    unknown = sorted(set(slides) - set(KIDNEY_SLIDES))
    if unknown:
        raise ValueError(f"Unknown kidney slide IDs: {unknown}")
    return slides


def fold_config(args: argparse.Namespace) -> dict:
    if args.protocol == "multi_loso":
        if not args.test_slide:
            raise ValueError("--test-slide is required for multi_loso")
        val_slide = args.val_slide or cyclic_val_slide(args.test_slide)
        if val_slide == args.test_slide:
            raise ValueError("--val-slide must differ from --test-slide")
        train_ids = [sid for sid in KIDNEY_SLIDES if sid not in {args.test_slide, val_slide}]
        test_slides = split_csv_slides(args.test_slides) or [args.test_slide]
        if test_slides != [args.test_slide]:
            raise ValueError("multi_loso predicts exactly its held-out --test-slide")
        tag = args.tag or f"test_{args.test_slide}_val_{val_slide}"
        return {
            "protocol": args.protocol,
            "tag": tag,
            "train_ids": train_ids,
            "val_ids": [val_slide],
            "test_slides": test_slides,
            "test_slide": args.test_slide,
            "val_slide": val_slide,
        }

    if not args.train_slide:
        raise ValueError("--train-slide is required for single_rotating")
    test_slides = split_csv_slides(args.test_slides) or [sid for sid in KIDNEY_SLIDES if sid != args.train_slide]
    if args.train_slide in test_slides:
        raise ValueError("--test-slides must not include --train-slide")
    tag = args.tag or f"train_{args.train_slide}_to_{'_'.join(test_slides)}"
    return {
        "protocol": args.protocol,
        "tag": tag,
        "train_ids": [args.train_slide],
        "val_ids": [],
        "test_slides": test_slides,
        "train_slide": args.train_slide,
        "val_slide": None,
    }


def checkpoint_dir(root: Path, protocol: str, tag: str) -> Path:
    return root / "checkpoints" / "kidney_hest" / "st_superres" / "protocol_rotating" / protocol / tag


def result_dir(root: Path, protocol: str, tag: str) -> Path:
    return root / "results" / "kidney_hest" / "morpho" / protocol / tag


def kidney_gene_setup(root: Path, ckpt_dir: Path, write_metadata: bool = True):
    from st_pipeline.data.gene_vocab import load_gene_vocab, map_genes_to_vocab

    geneset_path = (
        root
        / "data"
        / "spatial_data"
        / "genesets"
        / "kidney_hest_v1_INT13_INT14_INT15_INT17_INT18_INT19_INT21_INT24.txt"
    )
    if geneset_path.exists():
        common_genes = [g.strip() for g in geneset_path.read_text(encoding="utf-8").splitlines() if g.strip()]
    else:
        from st_pipeline.superres.feature_cache import get_common_genes

        common_genes = get_common_genes(KIDNEY_SLIDES, root=root)
    vocab = load_gene_vocab(root / "assets" / "cellfm" / "expand_gene_info.csv")
    kept_genes, gene_ids = map_genes_to_vocab(common_genes, vocab)
    if write_metadata:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "common_genes.json").write_text(json.dumps(kept_genes, indent=2), encoding="utf-8")
    return kept_genes, gene_ids, vocab


def ensure_kidney_cache(args, root: Path, slides: list[str]) -> None:
    core.require_torch()
    from st_pipeline.superres.feature_cache import ensure_feature_cache

    device = core.torch.device("cuda:0" if core.torch.cuda.is_available() else "cpu")
    ensure_feature_cache(
        slides,
        root=root,
        force=args.force_cache,
        device=device,
        max_dim=args.max_dim,
        use_tissue_seg=True,
    )


def build_datasets(root: Path, slides: list[str], kept_genes, gene_ids, device, force_cache: bool, max_dim: int):
    datasets = {}
    metas = {}
    for sid in slides:
        ds, meta = core.build_slide_dataset(
            root,
            sid,
            kept_genes,
            gene_ids,
            device,
            force_cache=force_cache,
            max_dim=max_dim,
            ensure_cache=force_cache,
        )
        datasets[sid] = ds
        metas[sid] = meta
    return datasets, metas


def training_dataset(args, root: Path, cfg: dict, kept_genes, gene_ids, device, is_ddp: bool, rank: int):
    from torch.utils.data import ConcatDataset, Subset

    train_ids = cfg["train_ids"]
    val_ids = cfg["val_ids"]
    needed = sorted(set(train_ids + val_ids))
    if rank == 0:
        ensure_kidney_cache(args, root, needed)
    if is_ddp:
        core.dist.barrier()
    datasets, metas = build_datasets(root, needed, kept_genes, gene_ids, device, False, args.max_dim)

    split_meta = None
    if cfg["protocol"] == "single_rotating":
        base = datasets[train_ids[0]]
        train_idx, val_idx, split_meta = core.make_blocked_split(base.locs_px, val_ratio=args.val_ratio, seed=args.seed)
        train_ds = Subset(base, train_idx)
        val_ds = Subset(base, val_idx)
        if rank == 0:
            meta = metas[train_ids[0]]
            split_meta = {
                **split_meta,
                "train_indices": train_idx,
                "val_indices": val_idx,
                "train_spot_ids": [meta["spot_ids"][i] for i in base.keep_idx[train_idx]],
                "val_spot_ids": [meta["spot_ids"][i] for i in base.keep_idx[val_idx]],
            }
    else:
        train_ds = ConcatDataset([datasets[sid] for sid in train_ids])
        val_ds = ConcatDataset([datasets[sid] for sid in val_ids])

    return train_ds, val_ds, datasets, metas, split_meta


def write_training_metadata(root: Path, ckpt_dir: Path, cfg: dict, metas: dict, split_meta: dict | None) -> None:
    for sid, meta in metas.items():
        np.savez(
            ckpt_dir / f"stats_{sid}.npz",
            radius_px=meta["radius_px"],
            mask_size=meta["mask_size"],
            embs_mean=meta["embs_mean"],
            embs_std=meta["embs_std"],
        )
    payload = {
        "protocol": cfg["protocol"],
        "slide_ids_all": KIDNEY_SLIDES,
        "train_ids": cfg["train_ids"],
        "val_ids": cfg["val_ids"],
        "test_slides": cfg["test_slides"],
        "tag": cfg["tag"],
    }
    if cfg["protocol"] == "multi_loso":
        payload["validation_strategy"] = "cyclic_next_slide"
        payload["heldout_test_slide"] = cfg["test_slide"]
        payload["validation_slide"] = cfg["val_slide"]
    else:
        payload["validation_strategy"] = "blocked_grid_quadrant"
        payload["validation_split"] = split_meta
    (ckpt_dir / "split.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compact_split_for_log(split_meta: dict | None) -> dict | None:
    if split_meta is None:
        return None
    keys = ["strategy", "grid_size", "block_count", "val_ratio_target", "val_ratio_actual", "val_blocks"]
    out = {key: split_meta[key] for key in keys if key in split_meta}
    if "train_indices" in split_meta:
        out["n_train_indices"] = len(split_meta["train_indices"])
    if "val_indices" in split_meta:
        out["n_val_indices"] = len(split_meta["val_indices"])
    if "train_spot_ids" in split_meta:
        out["n_train_spot_ids"] = len(split_meta["train_spot_ids"])
    if "val_spot_ids" in split_meta:
        out["n_val_spot_ids"] = len(split_meta["val_spot_ids"])
    return out


def train_fold(args, root: Path, cfg: dict) -> bool:
    core.require_torch()
    from st_pipeline.constants import KEYS
    from st_pipeline.data.collate import mil_collate
    from st_pipeline.model.nb_loss import nb_negative_log_likelihood

    torch = core.torch
    ckpt_dir = checkpoint_dir(root, cfg["protocol"], cfg["tag"])
    core.seed_everything(args.seed)
    is_ddp, rank, world_size, local_rank = core.init_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and is_ddp else "cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if rank == 0:
        print(f"ROOT: {root}")
        print(f"Training kidney fold {cfg['protocol']}::{cfg['tag']}")
        print(f"Train: {cfg['train_ids']} | Val: {cfg['val_ids']} | Test: {cfg['test_slides']}")
        print(f"Device: {device} | DDP={is_ddp} rank={rank}/{world_size}")

    kept_genes, gene_ids, vocab = kidney_gene_setup(root, ckpt_dir, write_metadata=(rank == 0))
    train_ds, val_ds, datasets, metas, split_meta = training_dataset(args, root, cfg, kept_genes, gene_ids, device, is_ddp, rank)
    if rank == 0:
        write_training_metadata(root, ckpt_dir, cfg, metas, split_meta)
        print(f"Train spots: {len(train_ds)} | Val spots: {len(val_ds)}")
        if split_meta is not None:
            print(f"Validation split: {compact_split_for_log(split_meta)}")

    train_sampler = core.DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    val_sampler = core.DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None
    train_loader = core.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=mil_collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = core.DataLoader(
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
        model = core.DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
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
        print(f"Genes: {len(kept_genes)} | chunks: {len(chunks)} | batch_size={args.batch_size}")

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        total = 0.0
        steps = 0
        for batch_idx, batch in enumerate(train_loader):
            for key in batch:
                batch[key] = batch[key].to(device, non_blocking=True)
            for ci in torch.randperm(len(chunks)).tolist():
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

    should_continue = rank == 0
    if is_ddp:
        core.dist.destroy_process_group()
    return should_continue


def predict_fold(args, root: Path, cfg: dict) -> None:
    core.require_torch()
    from st_pipeline.constants import KEYS
    from st_pipeline.data.collate import mil_collate
    from st_pipeline.data.gene_vocab import load_gene_vocab, map_genes_to_vocab

    torch = core.torch
    core.seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    ckpt_dir = checkpoint_dir(root, cfg["protocol"], cfg["tag"])
    out_root = result_dir(root, cfg["protocol"], cfg["tag"])
    out_root.mkdir(parents=True, exist_ok=True)

    kept_genes = json.loads((ckpt_dir / "common_genes.json").read_text(encoding="utf-8"))
    vocab = load_gene_vocab(root / "assets" / "cellfm" / "expand_gene_info.csv")
    kept_genes, gene_ids = map_genes_to_vocab(kept_genes, vocab)
    chunks = core.gene_chunks(len(kept_genes), args.gene_chunk_size)

    rows = []
    model = None
    input_dim = None
    for sid in cfg["test_slides"]:
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
        ds, meta = core.build_slide_dataset(root, sid, kept_genes, gene_ids, device, args.force_cache, args.max_dim)
        loader = core.DataLoader(
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
            ckpt_path = ckpt_dir / "best_model.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
        assert model is not None

        pred = np.zeros((len(ds), len(kept_genes)), dtype=np.float32)
        true = np.zeros((len(ds), len(kept_genes)), dtype=np.float32)
        offset = 0
        with torch.inference_mode():
            for b_idx, batch in enumerate(loader):
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
                if b_idx % args.heartbeat_batches == 0:
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


def aggregate_fold(root: Path, cfg: dict) -> None:
    out_root = result_dir(root, cfg["protocol"], cfg["tag"])
    rows = []
    for sid in cfg["test_slides"]:
        metrics_path = out_root / sid / "metrics.json"
        row = {
            "method": "Morpho-FM",
            "protocol": cfg["protocol"],
            "tag": cfg["tag"],
            "train_ids": ",".join(cfg["train_ids"]),
            "val_ids": ",".join(cfg["val_ids"]),
            "test_slide_id": sid,
            "result_dir": str(out_root / sid),
            "checkpoint_dir": str(checkpoint_dir(root, cfg["protocol"], cfg["tag"])),
            "status": "complete" if metrics_path.exists() else "missing_metrics",
        }
        if metrics_path.exists():
            row.update(json.loads(metrics_path.read_text(encoding="utf-8")))
        rows.append(row)
    summary_root = root / "results" / "kidney_hest" / "morpho" / "protocol_rotating"
    summary_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "fold_status.csv", index=False)
    master_path = summary_root / f"{cfg['protocol']}_{cfg['tag']}_summary.csv"
    df.to_csv(master_path, index=False)
    print(f"Saved fold status: {out_root / 'fold_status.csv'}")
    print(f"Saved fold summary: {master_path}")


def bench_fold(args, root: Path, cfg: dict) -> None:
    core.require_torch()
    from st_pipeline.constants import KEYS
    from st_pipeline.data.collate import mil_collate
    from st_pipeline.model.nb_loss import nb_negative_log_likelihood

    torch = core.torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kept_genes, gene_ids, vocab = kidney_gene_setup(root, checkpoint_dir(root, cfg["protocol"], cfg["tag"]), write_metadata=False)
    ensure_kidney_cache(args, root, sorted(set(cfg["train_ids"] + cfg["val_ids"])))
    datasets, _ = build_datasets(root, cfg["train_ids"], kept_genes, gene_ids, device, False, args.max_dim)
    from torch.utils.data import ConcatDataset

    ds = datasets[cfg["train_ids"][0]] if len(cfg["train_ids"]) == 1 else ConcatDataset([datasets[sid] for sid in cfg["train_ids"]])
    batch_sizes = [int(x) for x in args.bench_batch_sizes.split(",") if x.strip()]
    chunk_sizes = [int(x) for x in args.bench_gene_chunk_sizes.split(",") if x.strip()]
    rows = []
    for bs in batch_sizes:
        for chunk_size in chunk_sizes:
            try:
                torch.cuda.empty_cache()
                loader = core.DataLoader(ds, batch_size=bs, shuffle=True, num_workers=args.num_workers, collate_fn=mil_collate)
                sample_batch = next(iter(loader))
                input_dim = sample_batch[KEYS.X].shape[-1]
                model = core.build_model(input_dim, len(kept_genes), len(vocab), root, device)
                optimizer = torch.optim.AdamW(
                    [
                        {"params": [model.cellfm.model.gene_emb], "lr": args.lr * args.gene_lr_mult},
                        {"params": model.cellfm.model.cellwise_dec.map.parameters(), "lr": args.lr * args.gene_lr_mult},
                        {"params": model.adapter.parameters(), "lr": args.lr},
                        {"params": [model.gene_dispersion], "lr": args.lr},
                    ],
                    weight_decay=1e-2,
                )
                chunks = core.gene_chunks(len(kept_genes), chunk_size)
                use_amp = bool(args.use_amp and device.type == "cuda")
                scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
                start = time.perf_counter()
                steps = 0
                model.train()
                for batch_idx, batch in enumerate(loader):
                    if batch_idx >= args.probe_batches:
                        break
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
                        optimizer.zero_grad(set_to_none=True)
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            mu_bag, _ = model(batch_chunk)
                            theta = torch.exp(model.gene_dispersion[idx])
                            loss = nb_negative_log_likelihood(batch_chunk[KEYS.Y_BAG], mu_bag, theta)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    steps += 1
                elapsed = time.perf_counter() - start
                max_mem_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
                status = "ok"
                error = ""
            except Exception as exc:
                elapsed = float("nan")
                steps = 0
                max_mem_gb = float("nan")
                status = "failed"
                error = repr(exc)
            rows.append(
                {
                    "protocol": cfg["protocol"],
                    "tag": cfg["tag"],
                    "batch_size": bs,
                    "gene_chunk_size": chunk_size,
                    "probe_batches": args.probe_batches,
                    "steps_completed": steps,
                    "elapsed_sec": elapsed,
                    "sec_per_batch": elapsed / max(steps, 1) if status == "ok" else float("nan"),
                    "max_cuda_mem_gb": max_mem_gb,
                    "status": status,
                    "error": error,
                }
            )
            print(rows[-1])
    out_dir = root / "results" / "kidney_hest" / "morpho" / "batch_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{cfg['protocol']}_{cfg['tag']}_batch_probe.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved batch probe: {out}")


def main() -> None:
    args = parse_args()
    root = setup_root(args.root)
    cfg = fold_config(args)

    if args.mode == "cache":
        slides = sorted(set(cfg["train_ids"] + cfg["val_ids"] + cfg["test_slides"]))
        ensure_kidney_cache(args, root, slides)
        return
    if args.mode == "bench":
        bench_fold(args, root, cfg)
        return

    continue_after_train = True
    if args.mode in {"train", "all"}:
        continue_after_train = train_fold(args, root, cfg)
    if not continue_after_train:
        return
    if args.mode in {"predict", "all"}:
        predict_fold(args, root, cfg)
    if args.mode in {"aggregate", "all", "predict"}:
        aggregate_fold(root, cfg)


if __name__ == "__main__":
    main()
