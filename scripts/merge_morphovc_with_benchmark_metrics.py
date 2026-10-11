"""Merge local baseline benchmark metrics with Morpho-VC/Morpho-FM metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/share/home/nxy1zll2hjj/hjj/Morpho-VC"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "benchmark_audit")
    return parser.parse_args()


def to_local_result_dir(value: object, root: Path) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    if text.startswith(REMOTE_ROOT):
        rel = text[len(REMOTE_ROOT) :].lstrip("/")
        return str(root / Path(*rel.split("/")))
    return text


def standardize_metrics(df: pd.DataFrame, cohort: str, root: Path) -> pd.DataFrame:
    if df.empty:
        return df
    out = pd.DataFrame(index=df.index)
    out["cohort"] = cohort
    out["method"] = "Morpho-FM"
    if cohort == "Prostate":
        out["train_slide"] = df["train_slide"].astype(str)
        out["test_slide"] = df["test_slide_id"].astype(str)
        out["fold_tag"] = out["train_slide"] + "_to_" + out["test_slide"]
        out["priority_group"] = "morphofm_single_slide_rotating"
    else:
        train_col = "train_ids" if "train_ids" in df.columns else "split_train_ids"
        out["train_slide"] = df[train_col].astype(str)
        out["test_slide"] = df["test_slide_id"].astype(str)
        out["fold_tag"] = df["tag"].astype(str)
        out["priority_group"] = "morphofm_single_slide_rotating"
    out["result_dir"] = df["result_dir"].map(lambda v: to_local_result_dir(v, root))

    for col in [
        "n_spots",
        "n_genes",
        "MAE",
        "RMSE",
        "Mean_Pearson",
        "Median_Pearson",
        "HVG1000_Mean_Pearson",
        "HVG1000_Median_Pearson",
        "HVG1000_Overlap",
        "Best_Pearson",
    ]:
        out[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.NA

    if "Best_Gene_Index" in df.columns:
        out["Best_Gene_Index"] = pd.to_numeric(df["Best_Gene_Index"], errors="coerce")
    else:
        out["Best_Gene_Index"] = pd.NA
    return out


def load_prostate(root: Path) -> pd.DataFrame:
    protocol_dir = root / "results" / "prostate_hest" / "protocol_b_rotating"
    frames = []
    for path in sorted(protocol_dir.glob("INT*_morpho_summary.csv")):
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[df.get("status", "complete").eq("complete")].copy()
    return standardize_metrics(df, "Prostate", root)


def load_kidney(root: Path) -> pd.DataFrame:
    path = root / "results" / "kidney_hest" / "morpho" / "protocol_rotating" / "kidney_protocol_morpho_complete_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df.get("status", "complete").eq("complete")].copy()
    return standardize_metrics(df, "Kidney", root)


def method_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        metrics.groupby(["cohort", "method"], dropna=False)
        .agg(
            rows=("Mean_Pearson", "size"),
            mean_pearson=("Mean_Pearson", "mean"),
            median_pearson=("Median_Pearson", "mean"),
            hvg1000_mean=("HVG1000_Mean_Pearson", "mean"),
            mae=("MAE", "mean"),
            rmse=("RMSE", "mean"),
        )
        .reset_index()
    )
    numeric_cols = ["mean_pearson", "median_pearson", "hvg1000_mean", "mae", "rmse"]
    grouped[numeric_cols] = grouped[numeric_cols].round(4)
    return grouped.sort_values(["cohort", "mean_pearson"], ascending=[True, False])


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = out_dir / "benchmark_unified_metrics.csv"
    baseline = pd.read_csv(baseline_path)
    morpho = pd.concat([load_prostate(root), load_kidney(root)], ignore_index=True)
    if morpho.empty:
        raise SystemExit("No Morpho-FM metrics found.")

    combined = pd.concat([baseline, morpho], ignore_index=True, sort=False)
    combined_path = out_dir / "benchmark_unified_metrics_with_morphofm.csv"
    summary_path = out_dir / "benchmark_method_summary_with_morphofm.csv"
    morpho_path = out_dir / "morphofm_single_rotating_metrics.csv"

    combined.to_csv(combined_path, index=False)
    morpho.to_csv(morpho_path, index=False)
    summary = method_summary(combined)
    summary.to_csv(summary_path, index=False)

    print(f"Morpho-FM rows: {len(morpho)}")
    print(morpho.groupby('cohort').size().to_string())
    print(f"saved {morpho_path}")
    print(f"saved {combined_path}")
    print(f"saved {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
