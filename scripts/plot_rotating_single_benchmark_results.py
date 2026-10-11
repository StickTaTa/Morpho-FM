"""Create rotating single-slide benchmark figures from unified metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = Path(r"C:\Users\Administrator\Desktop\人生的第4篇论文")
DEFAULT_METRICS = ROOT / "results" / "benchmark_audit" / "benchmark_unified_metrics_with_morphofm.csv"
DEFAULT_REPO_OUT = ROOT / "results" / "paper_figures" / "rotating_single_benchmark"
DEFAULT_PAPER_OUT = PAPER_DIR / "Figure" / "benchmark_rebuild" / "rotating_single_benchmark"

METHOD_ORDER = ["HisToGene", "iStar", "mclSTExp", "sCellST", "HiST", "Morpho-FM"]
METHOD_COLORS = {
    "HisToGene": "#9A9A9A",
    "iStar": "#567D96",
    "mclSTExp": "#7E8F54",
    "sCellST": "#B47B5A",
    "HiST": "#8E6C8A",
    "Morpho-FM": "#C43C39",
}
COHORT_ORDER = ["Prostate", "Kidney"]
PROSTATE_SLIDES = ["INT25", "INT26", "INT27", "INT28"]
KIDNEY_SLIDES = ["INT13", "INT14", "INT15", "INT17", "INT18", "INT19", "INT21", "INT24"]
TEXT = "#202020"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--repo-out", type=Path, default=DEFAULT_REPO_OUT)
    parser.add_argument("--paper-out", type=Path, default=DEFAULT_PAPER_OUT)
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.65,
            "axes.edgecolor": TEXT,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
        }
    )


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["cohort"].isin(COHORT_ORDER) & df["method"].isin(METHOD_ORDER)].copy()
    for col in ["Mean_Pearson", "Median_Pearson", "HVG1000_Mean_Pearson", "MAE", "RMSE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["cohort"] = pd.Categorical(df["cohort"], COHORT_ORDER, ordered=True)
    df["method"] = pd.Categorical(df["method"], METHOD_ORDER, ordered=True)
    return df.sort_values(["cohort", "method", "train_slide", "test_slide"])


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["cohort", "method"], observed=True)
        .agg(
            n=("Mean_Pearson", "size"),
            mean_pearson=("Mean_Pearson", "mean"),
            sem_pearson=("Mean_Pearson", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
            median_pearson=("Median_Pearson", "mean"),
            hvg1000_mean=("HVG1000_Mean_Pearson", "mean"),
            mae=("MAE", "mean"),
            rmse=("RMSE", "mean"),
        )
        .reset_index()
    )
    return out


def save_tables(df: pd.DataFrame, outdirs: list[Path]) -> None:
    summary = summary_table(df)
    split = (
        df.groupby(["cohort", "method", "train_slide", "test_slide"], observed=True)
        .agg(
            mean_pearson=("Mean_Pearson", "mean"),
            median_pearson=("Median_Pearson", "mean"),
            hvg1000_mean=("HVG1000_Mean_Pearson", "mean"),
            mae=("MAE", "mean"),
            rmse=("RMSE", "mean"),
        )
        .reset_index()
    )
    for outdir in outdirs:
        outdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(outdir / "rotating_single_all_metrics.csv", index=False)
        summary.to_csv(outdir / "rotating_single_method_summary.csv", index=False)
        split.to_csv(outdir / "rotating_single_split_metrics.csv", index=False)


def draw_distribution_panel(ax: plt.Axes, df: pd.DataFrame, cohort: str, ylabel: str) -> None:
    sub = df[df["cohort"].astype(str).eq(cohort)].copy()
    rng = np.random.default_rng(7)
    for i, method in enumerate(METHOD_ORDER):
        vals = sub[sub["method"].astype(str).eq(method)]["Mean_Pearson"].dropna().to_numpy()
        if vals.size == 0:
            continue
        color = METHOD_COLORS[method]
        ax.boxplot(
            vals,
            positions=[i],
            widths=0.46,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            boxprops={"facecolor": color, "alpha": 0.18, "edgecolor": color, "linewidth": 0.8},
            medianprops={"color": TEXT, "linewidth": 0.9},
            whiskerprops={"color": color, "linewidth": 0.7},
            capprops={"color": color, "linewidth": 0.7},
        )
        jitter = rng.normal(0, 0.055, size=vals.size)
        ax.scatter(np.full(vals.size, i) + jitter, vals, s=13, color=color, alpha=0.72, linewidths=0)
        ax.scatter([i], [vals.mean()], s=32, color=color, edgecolor="white", linewidth=0.8, zorder=4)
    ax.set_title(cohort, loc="left", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels(METHOD_ORDER, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis="y", color="#E2E2E2", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=7)


def plot_method_distribution(df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(174 / 25.4, 78 / 25.4), sharey=True)
    for ax, cohort in zip(axes, COHORT_ORDER):
        draw_distribution_panel(ax, df, cohort, "Mean per-gene Pearson r")
        n_map = df[df["cohort"].astype(str).eq(cohort)].groupby("method", observed=True).size()
        counts = ", ".join(f"{m} {int(n_map.get(m, 0))}" for m in METHOD_ORDER)
        ax.text(0, -0.38, counts, transform=ax.transAxes, fontsize=5.7, color="#666666", ha="left")
    axes[0].set_ylim(0.0, max(0.42, float(df["Mean_Pearson"].max()) + 0.035))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.34, top=0.88, wspace=0.18)
    fig.savefig(outdir / "main_rotating_single_method_distribution.pdf", bbox_inches="tight")
    fig.savefig(outdir / "main_rotating_single_method_distribution.png", dpi=420, bbox_inches="tight")
    plt.close(fig)


def matrix_for_morpho(df: pd.DataFrame, cohort: str, slides: list[str]) -> pd.DataFrame:
    sub = df[(df["cohort"].astype(str).eq(cohort)) & (df["method"].astype(str).eq("Morpho-FM"))]
    mat = pd.DataFrame(np.nan, index=slides, columns=slides)
    for _, row in sub.iterrows():
        train = str(row["train_slide"])
        test = str(row["test_slide"])
        if train in mat.index and test in mat.columns:
            mat.loc[train, test] = row["Mean_Pearson"]
    return mat


def heatmap(ax: plt.Axes, mat: pd.DataFrame, title: str, vmin: float, vmax: float) -> None:
    cmap = LinearSegmentedColormap.from_list("morphofm_heat", ["#F3F3F3", "#A7C7C9", "#2D7488", "#C43C39"])
    cmap.set_bad("#EFEFEF")
    arr = mat.to_numpy(dtype=float)
    im = ax.imshow(np.ma.masked_invalid(arr), vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_xticklabels(mat.columns, fontsize=7)
    ax.set_yticklabels(mat.index, fontsize=7)
    ax.tick_params(axis="x", rotation=45, length=0, pad=1.5)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_xlabel("Test slide", fontsize=7.5)
    ax.set_ylabel("Training slide", fontsize=7.5)
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold", pad=6)
    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = arr[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6, color=("white" if value > 0.25 else TEXT))
    return im


def plot_morphofm_heatmaps(df: pd.DataFrame, outdir: Path) -> None:
    kidney = matrix_for_morpho(df, "Kidney", KIDNEY_SLIDES)
    prostate = matrix_for_morpho(df, "Prostate", PROSTATE_SLIDES)
    fig = plt.figure(figsize=(174 / 25.4, 84 / 25.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 0.82, 0.05], wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    im = heatmap(ax1, kidney, "a  Kidney Morpho-FM", 0.10, 0.38)
    heatmap(ax2, prostate, "b  Prostate Morpho-FM", 0.10, 0.38)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=6.5, width=0.5)
    cbar.set_label("Mean Pearson r", fontsize=7)
    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.18, top=0.88)
    fig.savefig(outdir / "supp_morphofm_rotating_single_heatmaps.pdf", bbox_inches="tight")
    fig.savefig(outdir / "supp_morphofm_rotating_single_heatmaps.png", dpi=420, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_style()
    df = load_metrics(args.metrics)
    outdirs = [args.repo_out, args.paper_out]
    for outdir in outdirs:
        outdir.mkdir(parents=True, exist_ok=True)
        plot_method_distribution(df, outdir)
        plot_morphofm_heatmaps(df, outdir)
    save_tables(df, outdirs)
    summary = summary_table(df).sort_values(["cohort", "mean_pearson"], ascending=[True, False])
    print(f"saved: {args.repo_out}")
    print(f"saved: {args.paper_out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
