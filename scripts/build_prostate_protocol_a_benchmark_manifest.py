from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "benchmark_audit"

METHODS = {
    "HisToGene": "histogene",
    "iStar": "istar",
    "mclSTExp": "mclstexp",
    "sCellST": "scellst",
    "HiST": "hist",
    "Path2Space": "path2space",
    "ST-Net": "stnet",
    "STimage": "stimage",
}

FOLDS = [
    {"tag": "test_INT25_val_INT26", "train_slides": ["INT27", "INT28"], "val_slide": "INT26", "test_slide": "INT25"},
    {"tag": "test_INT26_val_INT27", "train_slides": ["INT25", "INT28"], "val_slide": "INT27", "test_slide": "INT26"},
    {"tag": "test_INT27_val_INT28", "train_slides": ["INT25", "INT26"], "val_slide": "INT28", "test_slide": "INT27"},
    {"tag": "test_INT28_val_INT25", "train_slides": ["INT26", "INT27"], "val_slide": "INT25", "test_slide": "INT28"},
]

METRIC_FIELDS = [
    "n_spots",
    "n_genes",
    "MAE",
    "RMSE",
    "Mean_Pearson",
    "Median_Pearson",
    "HVG1000_Mean_Pearson",
    "HVG1000_Median_Pearson",
    "HVG1000_Overlap",
    "Best_Gene",
    "Best_Pearson",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prostate Protocol-A multi-slide benchmark manifest.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"metrics_read_error": repr(exc)}
    return {key: payload.get(key) for key in METRIC_FIELDS if key in payload}


def status_for(result_dir: Path) -> str:
    has_metrics = (result_dir / "metrics.json").exists()
    has_pred = (result_dir / "pred_bag.npy").exists()
    has_true = (result_dir / "true_bag.npy").exists()
    if has_metrics and has_pred and has_true:
        return "complete"
    if has_metrics:
        return "metrics_only"
    if has_pred or has_true:
        return "partial_prediction"
    return "missing"


def build_manifest(root: Path) -> pd.DataFrame:
    rows = []
    for method, method_dir in METHODS.items():
        for fold in FOLDS:
            result_dir = root / "results" / "prostate_hest" / method_dir / "multi_rotating" / fold["tag"] / fold["test_slide"]
            row = {
                "cohort": "Prostate",
                "cohort_dir": "prostate_hest",
                "method": method,
                "method_dir": method_dir,
                "protocol": "rotating_heldout_multi_slide",
                "fold_tag": fold["tag"],
                "train_slides": ",".join(fold["train_slides"]),
                "val_slide": fold["val_slide"],
                "test_slide": fold["test_slide"],
                "status": status_for(result_dir),
                "metrics_exists": (result_dir / "metrics.json").exists(),
                "pred_exists": (result_dir / "pred_bag.npy").exists(),
                "true_exists": (result_dir / "true_bag.npy").exists(),
                "unified_metrics_exists": (result_dir / "unified_metrics.json").exists(),
                "result_dir": str(result_dir),
                "metrics_path": str(result_dir / "metrics.json"),
                "pred_path": str(result_dir / "pred_bag.npy"),
                "true_path": str(result_dir / "true_bag.npy"),
            }
            row.update(read_metrics(result_dir / "metrics.json"))
            rows.append(row)
    return pd.DataFrame(rows)


def build_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    for col in ["n_genes"]:
        if col not in manifest.columns:
            manifest[col] = pd.NA
    summary = (
        manifest.groupby(["method"], dropna=False)
        .agg(
            expected=("status", "size"),
            complete=("status", lambda s: int((s == "complete").sum())),
            metrics_only=("status", lambda s: int((s == "metrics_only").sum())),
            partial_prediction=("status", lambda s: int((s == "partial_prediction").sum())),
            missing=("status", lambda s: int((s == "missing").sum())),
            n_gene_min=("n_genes", "min"),
            n_gene_max=("n_genes", "max"),
        )
        .reset_index()
    )
    summary["complete_fraction"] = summary["complete"] / summary["expected"]
    return summary


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(root)
    summary = build_summary(manifest)
    missing = manifest[manifest["status"].ne("complete")].copy()

    manifest_path = out_dir / "prostate_protocol_a_benchmark_manifest.csv"
    summary_path = out_dir / "prostate_protocol_a_benchmark_summary.csv"
    missing_path = out_dir / "prostate_protocol_a_benchmark_missing.csv"
    manifest.to_csv(manifest_path, index=False)
    summary.to_csv(summary_path, index=False)
    missing.to_csv(missing_path, index=False)

    print(f"[saved] {manifest_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {missing_path}")
    print(summary[["method", "expected", "complete", "missing", "complete_fraction"]])


if __name__ == "__main__":
    main()
