"""Summarize Morpho-FM advantage over the strongest baseline per cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = ROOT / "results" / "benchmark_audit" / "benchmark_unified_metrics_with_morphofm.csv"
DEFAULT_OUT = ROOT / "results" / "benchmark_audit" / "benchmark_morphofm_vs_best_baseline_stats.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def wilcoxon_pvalue(diff: np.ndarray) -> float:
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(diff, zero_method="wilcox", alternative="greater").pvalue)
    except Exception:
        return float("nan")


def bootstrap_ci(diff: np.ndarray, seed: int = 13, n_boot: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).astype(float))


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.metrics)
    rows = []
    for cohort, sub in df.groupby("cohort"):
        method_mean = (
            sub[sub["method"].ne("Morpho-FM")]
            .groupby("method")["Mean_Pearson"]
            .mean()
            .sort_values(ascending=False)
        )
        best = str(method_mean.index[0])
        key_cols = ["train_slide", "test_slide"]
        morpho = sub[sub["method"].eq("Morpho-FM")][key_cols + ["Mean_Pearson"]].rename(
            columns={"Mean_Pearson": "morphofm"}
        )
        baseline = sub[sub["method"].eq(best)][key_cols + ["Mean_Pearson"]].rename(
            columns={"Mean_Pearson": "best_baseline"}
        )
        paired = morpho.merge(baseline, on=key_cols, how="inner")
        diff = (paired["morphofm"] - paired["best_baseline"]).to_numpy(dtype=float)
        lo, hi = bootstrap_ci(diff)
        rows.append(
            {
                "cohort": cohort,
                "best_baseline": best,
                "paired_n": len(diff),
                "morphofm_mean": paired["morphofm"].mean(),
                "best_baseline_mean": paired["best_baseline"].mean(),
                "mean_difference": diff.mean(),
                "median_difference": np.median(diff),
                "bootstrap95_low": lo,
                "bootstrap95_high": hi,
                "wilcoxon_greater_p": wilcoxon_pvalue(diff),
                "all_directions_positive": bool(np.all(diff > 0)),
            }
        )
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
