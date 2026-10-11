from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results" / "benchmark_audit" / "benchmark_task_manifest.csv"
DEFAULT_OUT = ROOT / "results" / "benchmark_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute unified metrics for benchmark prediction matrices.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true", help="Overwrite existing unified_metrics.json files.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick checks.")
    parser.add_argument("--methods", nargs="*", default=None, help="Optional method labels to evaluate.")
    parser.add_argument("--cohorts", nargs="*", default=None, help="Optional cohort labels to evaluate.")
    return parser.parse_args()


def safe_pearson_vec(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    true = true.astype(np.float64, copy=False)
    pred = pred.astype(np.float64, copy=False)
    true_centered = true - true.mean(axis=0, keepdims=True)
    pred_centered = pred - pred.mean(axis=0, keepdims=True)
    numerator = (true_centered * pred_centered).sum(axis=0)
    denominator = np.sqrt((true_centered * true_centered).sum(axis=0) * (pred_centered * pred_centered).sum(axis=0))
    out = np.full(true.shape[1], np.nan, dtype=np.float32)
    valid = denominator > 0
    out[valid] = (numerator[valid] / denominator[valid]).astype(np.float32)
    return out


def metric_payload(pred_path: Path, true_path: Path, out_dir: Path) -> dict:
    pred = np.load(pred_path)
    true = np.load(true_path)
    if pred.shape != true.shape:
        n_spots = min(pred.shape[0], true.shape[0])
        n_genes = min(pred.shape[1], true.shape[1])
        pred = pred[:n_spots, :n_genes]
        true = true[:n_spots, :n_genes]

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

    np.save(out_dir / "unified_gene_corr_all.npy", corr_all)
    np.save(out_dir / "unified_gene_corr_hvg.npy", corr_hvg)
    np.save(out_dir / "unified_hvg_idx_true.npy", hvg_idx_true)
    np.save(out_dir / "unified_hvg_idx_pred.npy", hvg_idx_pred)
    np.save(out_dir / "unified_corr_overlap_genes.npy", corr_overlap)

    return {
        "n_spots": int(pred.shape[0]),
        "n_genes": int(pred.shape[1]),
        "MAE": float(np.mean(np.abs(pred - true))),
        "RMSE": float(np.sqrt(np.mean((pred - true) ** 2))),
        "Mean_Pearson": float(np.nanmean(corr_all)) if valid.size else float("nan"),
        "Median_Pearson": float(np.nanmedian(corr_all)) if valid.size else float("nan"),
        "HVG1000_Mean_Pearson": float(np.nanmean(corr_hvg)),
        "HVG1000_Median_Pearson": float(np.nanmedian(corr_hvg)),
        "HVG1000_Overlap": int(overlap_idx.size),
        "Best_Gene_Index": best_idx,
        "Best_Pearson": float(corr_all[best_idx]) if best_idx is not None else None,
    }


def filtered_manifest(args: argparse.Namespace) -> pd.DataFrame:
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["pred_exists"].eq(True) & manifest["true_exists"].eq(True)].copy()
    if args.methods:
        manifest = manifest[manifest["method"].isin(args.methods)]
    if args.cohorts:
        manifest = manifest[manifest["cohort"].isin(args.cohorts)]
    if args.limit is not None:
        manifest = manifest.head(args.limit)
    return manifest


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = filtered_manifest(args)
    rows: list[dict] = []
    for i, row in manifest.iterrows():
        result_dir = Path(row["result_dir"])
        unified_path = result_dir / "unified_metrics.json"
        if unified_path.exists() and not args.force:
            payload = json.loads(unified_path.read_text(encoding="utf-8"))
        else:
            payload = metric_payload(Path(row["pred_path"]), Path(row["true_path"]), result_dir)
            unified_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # The two manifests name the training side differently: the single-slide
        # manifest has `train_slide` (one id) while the Protocol-A manifest has
        # `train_slides` (two ids) plus `val_slide`, and only the single-slide manifest
        # carries `priority_group`. Read both defensively so either manifest can be
        # evaluated by this script instead of only the one it was written against.
        train_label = row.get("train_slide")
        if train_label is None or (isinstance(train_label, float) and pd.isna(train_label)):
            train_label = row.get("train_slides", "")
        rows.append(
            {
                "cohort": row["cohort"],
                "method": row["method"],
                "train_slide": train_label,
                "test_slide": row["test_slide"],
                "fold_tag": row["fold_tag"],
                "priority_group": row.get("priority_group", ""),
                "result_dir": row["result_dir"],
                **payload,
            }
        )
        print(f"[{len(rows)}/{len(manifest)}] {row['cohort']} {row['method']} {train_label}->{row['test_slide']}")

    metrics = pd.DataFrame(rows)
    metrics_path = out_dir / "benchmark_unified_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"[saved] {metrics_path}")


if __name__ == "__main__":
    main()
