from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import kidney_protocol_morpho as engine
import prostate_protocol_b_morpho as core


PROSTATE_SLIDES = ["INT25", "INT26", "INT27", "INT28"]
PROTOCOL = "prostate_protocol_a_rotating_multi_slide"
FOLD_TABLE = {
    "INT25": {"val": "INT26", "train": ["INT27", "INT28"]},
    "INT26": {"val": "INT27", "train": ["INT25", "INT28"]},
    "INT27": {"val": "INT28", "train": ["INT25", "INT26"]},
    "INT28": {"val": "INT25", "train": ["INT26", "INT27"]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Morpho-FM prostate Protocol A: rotating held-out multi-slide "
            "benchmark with two training slides, one validation slide and one test slide."
        )
    )
    parser.add_argument("--root", type=Path, default=None, help="Project root. Defaults to MORPHO_VC_ROOT or cwd.")
    parser.add_argument("--test-slide", choices=PROSTATE_SLIDES, required=True)
    parser.add_argument("--val-slide", choices=PROSTATE_SLIDES, default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--mode", choices=["train", "predict", "all", "aggregate"], default="all")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-predict", action="store_true")
    parser.add_argument("--resume-training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--predict-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gene-chunk-size", type=int, default=512)
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
    for path in (root / "src", root / "scripts"):
        if str(path) not in sys.path:
            sys.path.append(str(path))
    return root


def fold_config(args: argparse.Namespace) -> dict:
    val_slide = args.val_slide or FOLD_TABLE[args.test_slide]["val"]
    if val_slide == args.test_slide:
        raise ValueError("--val-slide must differ from --test-slide")
    train_ids = [sid for sid in PROSTATE_SLIDES if sid not in {args.test_slide, val_slide}]
    if len(train_ids) != 2:
        raise ValueError(f"Protocol A expects exactly two training slides, got {train_ids}")
    tag = args.tag or f"test_{args.test_slide}_val_{val_slide}"
    return {
        "protocol": PROTOCOL,
        "tag": tag,
        "train_ids": train_ids,
        "val_ids": [val_slide],
        "test_slides": [args.test_slide],
        "test_slide": args.test_slide,
        "val_slide": val_slide,
    }


def checkpoint_dir(root: Path, protocol: str, tag: str) -> Path:
    return root / "checkpoints" / "st_superres" / "protocol_a_rotating" / f"cellfm_multi_{tag}"


def result_dir(root: Path, protocol: str, tag: str) -> Path:
    return root / "results" / "prostate_hest" / "morpho" / "multi_rotating" / tag


def prostate_gene_setup(root: Path, ckpt_dir: Path, write_metadata: bool = True):
    return core.load_gene_setup(root, ckpt_dir, write_metadata=write_metadata)


def ensure_prostate_cache(args: argparse.Namespace, root: Path, slides: list[str]) -> None:
    core.require_torch()
    from st_pipeline.superres.feature_cache import ensure_feature_cache

    torch = core.torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ensure_feature_cache(
        slides,
        root=root,
        force=args.force_cache,
        device=device,
        max_dim=args.max_dim,
        use_tissue_seg=True,
    )


def write_training_metadata(root: Path, ckpt_dir: Path, cfg: dict, metas: dict, split_meta: dict | None) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for sid, meta in metas.items():
        np.savez(
            ckpt_dir / f"stats_{sid}.npz",
            radius_px=meta["radius_px"],
            mask_size=meta["mask_size"],
            embs_mean=meta["embs_mean"],
            embs_std=meta["embs_std"],
        )
    payload = {
        "protocol": PROTOCOL,
        "slide_ids_all": PROSTATE_SLIDES,
        "train_ids": cfg["train_ids"],
        "val_ids": cfg["val_ids"],
        "test_slides": cfg["test_slides"],
        "heldout_test_slide": cfg["test_slide"],
        "validation_slide": cfg["val_slide"],
        "validation_strategy": "heldout_slide",
        "tag": cfg["tag"],
    }
    (ckpt_dir / "split.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def patch_engine() -> None:
    engine.KIDNEY_SLIDES = PROSTATE_SLIDES
    engine.checkpoint_dir = checkpoint_dir
    engine.result_dir = result_dir
    engine.kidney_gene_setup = prostate_gene_setup
    engine.ensure_kidney_cache = ensure_prostate_cache
    engine.write_training_metadata = write_training_metadata


def aggregate_fold(root: Path, cfg: dict) -> None:
    out_root = result_dir(root, cfg["protocol"], cfg["tag"])
    rows = []
    for sid in cfg["test_slides"]:
        metrics_path = out_root / sid / "metrics.json"
        row = {
            "method": "Morpho-FM",
            "protocol": PROTOCOL,
            "tag": cfg["tag"],
            "train_slides": ",".join(cfg["train_ids"]),
            "val_slide": ",".join(cfg["val_ids"]),
            "test_slide_id": sid,
            "status": "complete" if metrics_path.exists() else "missing_metrics",
            "checkpoint_dir": str(checkpoint_dir(root, cfg["protocol"], cfg["tag"])),
            "result_dir": str(out_root / sid),
        }
        if metrics_path.exists():
            row.update(json.loads(metrics_path.read_text(encoding="utf-8")))
        rows.append(row)
    summary_root = root / "results" / "prostate_hest" / "morpho" / "protocol_a_rotating"
    summary_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "fold_status.csv", index=False)
    master_path = summary_root / f"{cfg['tag']}_morpho_summary.csv"
    df.to_csv(master_path, index=False)
    print(f"Saved fold status: {out_root / 'fold_status.csv'}")
    print(f"Saved fold summary: {master_path}")


def main() -> None:
    args = parse_args()
    root = setup_root(args.root)
    patch_engine()
    cfg = fold_config(args)
    continue_after_train = True
    if args.mode in {"train", "all"}:
        continue_after_train = engine.train_fold(args, root, cfg)
    if not continue_after_train:
        return
    if args.mode in {"predict", "all"}:
        engine.predict_fold(args, root, cfg)
    if args.mode in {"aggregate", "all", "predict"}:
        aggregate_fold(root, cfg)


if __name__ == "__main__":
    main()
