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

OPTIONAL_METHODS = {
    "THItoGene": "thitogene",
}

COHORTS = {
    "prostate_hest": {
        "label": "Prostate",
        "slides": ["INT25", "INT26", "INT27", "INT28"],
        "representative_train_slides": ["INT25", "INT26", "INT27", "INT28"],
    },
    "kidney_hest": {
        "label": "Kidney",
        "slides": ["INT13", "INT14", "INT15", "INT17", "INT18", "INT19", "INT21", "INT24"],
        "representative_train_slides": ["INT13", "INT18", "INT24"],
    },
}

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
    parser = argparse.ArgumentParser(description="Build expected benchmark task manifest for rotating single-slide baselines.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--include-thitogene", action="store_true", help="Include optional THItoGene rows.")
    return parser.parse_args()


def fold_tag(train_slide: str, slides: list[str]) -> tuple[str, list[str]]:
    test_slides = [sid for sid in slides if sid != train_slide]
    return f"{train_slide}_to_{'_'.join(test_slides)}", test_slides


def read_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"metrics_read_error": repr(exc)}
    return {key: payload.get(key) for key in METRIC_FIELDS if key in payload}


def status_for(metrics_path: Path, pred_path: Path, true_path: Path) -> str:
    has_metrics = metrics_path.exists()
    has_pred = pred_path.exists()
    has_true = true_path.exists()
    if has_metrics and has_pred and has_true:
        return "complete"
    if has_metrics:
        return "metrics_only"
    if has_pred or has_true:
        return "partial_prediction"
    return "missing"


def build_manifest(root: Path, include_thitogene: bool = False) -> pd.DataFrame:
    methods = dict(METHODS)
    if include_thitogene:
        methods.update(OPTIONAL_METHODS)

    rows: list[dict] = []
    for cohort, cohort_meta in COHORTS.items():
        slides = cohort_meta["slides"]
        representative = set(cohort_meta["representative_train_slides"])
        for method_label, method_dir in methods.items():
            for train_slide in slides:
                tag, test_slides = fold_tag(train_slide, slides)
                priority = "required_prostate" if cohort == "prostate_hest" else (
                    "kidney_representative" if train_slide in representative else "kidney_full"
                )
                for test_slide in test_slides:
                    result_dir = root / "results" / cohort / method_dir / "single" / tag / test_slide
                    metrics_path = result_dir / "metrics.json"
                    pred_path = result_dir / "pred_bag.npy"
                    true_path = result_dir / "true_bag.npy"
                    row = {
                        "cohort": cohort_meta["label"],
                        "cohort_dir": cohort,
                        "method": method_label,
                        "method_dir": method_dir,
                        "protocol": "rotating_single_slide",
                        "train_slide": train_slide,
                        "test_slide": test_slide,
                        "fold_tag": tag,
                        "priority_group": priority,
                        "status": status_for(metrics_path, pred_path, true_path),
                        "metrics_exists": metrics_path.exists(),
                        "pred_exists": pred_path.exists(),
                        "true_exists": true_path.exists(),
                        "gene_corr_exists": (result_dir / "gene_corr_all.npy").exists(),
                        "result_dir": str(result_dir),
                        "metrics_path": str(metrics_path),
                        "pred_path": str(pred_path),
                        "true_path": str(true_path),
                    }
                    row.update(read_metrics(metrics_path))
                    rows.append(row)
    return pd.DataFrame(rows)


def build_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    expected = (
        manifest.groupby(["cohort", "method", "priority_group"], dropna=False)
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
    expected["complete_fraction"] = expected["complete"] / expected["expected"]
    return expected


def build_legacy_results(root: Path, manifest: pd.DataFrame, include_thitogene: bool = False) -> pd.DataFrame:
    methods = dict(METHODS)
    if include_thitogene:
        methods.update(OPTIONAL_METHODS)
    expected_dirs = {Path(p).resolve() for p in manifest["result_dir"].astype(str)}
    rows = []
    for cohort, cohort_meta in COHORTS.items():
        for method_label, method_dir in methods.items():
            base = root / "results" / cohort / method_dir / "single"
            if not base.exists():
                continue
            for metrics_path in sorted(base.rglob("metrics.json")):
                result_dir = metrics_path.parent.resolve()
                if result_dir in expected_dirs:
                    continue
                fold_tag = metrics_path.parent.parent.name if metrics_path.parent.parent != base else metrics_path.parent.name
                test_slide = metrics_path.parent.name if metrics_path.parent.parent != base else None
                row = {
                    "cohort": cohort_meta["label"],
                    "cohort_dir": cohort,
                    "method": method_label,
                    "method_dir": method_dir,
                    "fold_tag": fold_tag,
                    "test_slide": test_slide,
                    "result_dir": str(result_dir),
                    "metrics_path": str(metrics_path),
                    "pred_exists": (result_dir / "pred_bag.npy").exists(),
                    "true_exists": (result_dir / "true_bag.npy").exists(),
                }
                row.update(read_metrics(metrics_path))
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(root, include_thitogene=args.include_thitogene)
    summary = build_summary(manifest)
    missing = manifest[manifest["status"].ne("complete")].copy()
    legacy = build_legacy_results(root, manifest, include_thitogene=args.include_thitogene)

    manifest_path = out_dir / "benchmark_task_manifest.csv"
    summary_path = out_dir / "benchmark_task_summary.csv"
    missing_path = out_dir / "benchmark_missing_tasks.csv"
    legacy_path = out_dir / "benchmark_legacy_single_results.csv"
    manifest.to_csv(manifest_path, index=False)
    summary.to_csv(summary_path, index=False)
    missing.to_csv(missing_path, index=False)
    legacy.to_csv(legacy_path, index=False)

    print(f"[saved] {manifest_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {missing_path}")
    print(f"[saved] {legacy_path}")
    print(summary[["cohort", "method", "priority_group", "expected", "complete", "missing", "complete_fraction"]])


if __name__ == "__main__":
    main()
