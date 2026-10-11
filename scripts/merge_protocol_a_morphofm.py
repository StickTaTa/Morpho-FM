"""Merge Morpho-FM into the Protocol-A unified table and emit a 9-method summary.

Why this exists
---------------
`results/benchmark_audit/protocol_a_unified/benchmark_unified_metrics.csv` is produced by
`scripts/evaluate_benchmark_predictions.py` from
`prostate_protocol_a_benchmark_manifest.csv`, which covers the **8 comparison baselines
only** — Morpho-FM is not in that manifest, because its Protocol-A runs are produced by a
different pipeline and summarised separately in
`results/prostate_hest/morpho/protocol_a_rotating/prostate_protocol_a_morpho_complete_metrics.csv`.

This script does not re-derive any metric. It re-shapes Morpho-FM's already-computed
numbers into the unified schema and concatenates them, so the multi-slide figures can be
drawn from one 9-method source.

Usage:
    python scripts/merge_protocol_a_morphofm.py
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UNIFIED = ROOT / "results" / "benchmark_audit" / "protocol_a_unified" / "benchmark_unified_metrics.csv"
MORPHO = (ROOT / "results" / "prostate_hest" / "morpho" / "protocol_a_rotating"
          / "prostate_protocol_a_morpho_complete_metrics.csv")
OUT = UNIFIED.with_name("benchmark_unified_metrics_with_morphofm.csv")
SUMMARY = UNIFIED.with_name("protocol_a_method_summary.csv")


def main() -> None:
    uni = pd.read_csv(UNIFIED)
    mor = pd.read_csv(MORPHO)

    # Re-shape Morpho-FM rows into the unified schema. No metric is recomputed.
    mor = pd.DataFrame({
        "cohort": "Prostate",
        "method": "Morpho-FM",
        "train_slide": mor["train_slides"],
        "test_slide": mor["test_slide_id"],
        "fold_tag": mor["tag"],
        "priority_group": "protocol_a_rotating",
        "result_dir": mor["result_dir"],
        "n_spots": mor["n_spots"],
        "n_genes": mor["n_genes"],
        "MAE": mor["MAE"],
        "RMSE": mor["RMSE"],
        "Mean_Pearson": mor["Mean_Pearson"],
        "Median_Pearson": mor["Median_Pearson"],
        "HVG1000_Mean_Pearson": mor["HVG1000_Mean_Pearson"],
        "HVG1000_Median_Pearson": mor["HVG1000_Median_Pearson"],
        "HVG1000_Overlap": mor["HVG1000_Overlap"],
        "Best_Gene_Index": pd.NA,
        "Best_Pearson": mor["Best_Pearson"],
    })

    both = pd.concat([uni, mor], ignore_index=True)
    both.to_csv(OUT, index=False)
    print(f"[saved] {OUT}  ({len(both)} rows, {both['method'].nunique()} methods)")

    summary = (
        both.groupby("method")
        .agg(
            n_folds=("Mean_Pearson", "size"),
            mean_pearson=("Mean_Pearson", "mean"),
            median_pearson=("Median_Pearson", "median"),
            hvg1000_mean=("HVG1000_Mean_Pearson", "mean"),
            mae=("MAE", "mean"),
            rmse=("RMSE", "mean"),
            n_genes_min=("n_genes", "min"),
            n_genes_max=("n_genes", "max"),
        )
        .sort_values("mean_pearson", ascending=False)
        .reset_index()
    )
    summary.to_csv(SUMMARY, index=False)
    print(f"[saved] {SUMMARY}")
    print()
    with pd.option_context("display.width", 200):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
