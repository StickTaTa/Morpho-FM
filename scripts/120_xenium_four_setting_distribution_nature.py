from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "st_superres" / "xenium_extended"
DEFAULT_OUTDIR = RESULT_ROOT / "panel" / "figures"

MM_TO_INCH = 1 / 25.4
FIGSIZE = (183 * MM_TO_INCH, 67 * MM_TO_INCH)

RUNS = [
    {
        "run_name": "XEN1_within",
        "label": "XEN1\n-> XEN1",
        "mode": "within",
    },
    {
        "run_name": "XEN2_to_XEN1",
        "label": "XEN2\n-> XEN1",
        "mode": "cross",
    },
    {
        "run_name": "XEN2_within",
        "label": "XEN2\n-> XEN2",
        "mode": "within",
    },
    {
        "run_name": "XEN1_to_XEN2",
        "label": "XEN1\n-> XEN2",
        "mode": "cross",
    },
]

METRICS = [
    {
        "key": "full_spot",
        "title": "306-gene spot",
        "letter": "a",
        "source": "full",
        "column": "spot_pearson",
        "n_label": "n = 306 genes",
    },
    {
        "key": "panel_spot",
        "title": "29-gene spot",
        "letter": "b",
        "source": "panel",
        "column": "spot_pearson",
        "n_label": "n = 29 genes",
    },
    {
        "key": "panel_dense",
        "title": "29-gene dense back-projection",
        "letter": "c",
        "source": "panel",
        "column": "dense_backproj_pearson",
        "n_label": "n = 29 genes",
    },
]

PALETTE = {
    "within": "#2B6F8A",
    "cross": "#C76B4A",
}
EDGE = {
    "within": "#174A5E",
    "cross": "#8D3E2C",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redraw the Xenium four-setting distribution panel in a compact journal style."
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=RESULT_ROOT,
        help="Root containing panel CSV outputs.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory for the redrawn figure.",
    )
    parser.add_argument(
        "--basename",
        default="xenium_four_setting_distribution_nature",
        help="Output file basename without suffix.",
    )
    return parser.parse_args()


def load_metric_values(result_root: Path) -> dict[str, list[np.ndarray]]:
    panel_root = result_root / "panel"
    full_df = pd.read_csv(panel_root / "xenium_306gene_spot_metrics_detail.csv")
    panel_df = pd.read_csv(panel_root / "xenium_balanced_panel_metrics_detail.csv")

    values: dict[str, list[np.ndarray]] = {}
    for metric in METRICS:
        source = full_df if metric["source"] == "full" else panel_df
        values[metric["key"]] = [
            source.loc[source["run_name"] == run["run_name"], metric["column"]]
            .dropna()
            .astype(float)
            .to_numpy()
            for run in RUNS
        ]
    return values


def style_axes(ax: plt.Axes, show_ylabel: bool) -> None:
    ax.set_ylim(-0.04, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.55)
    ax.axhline(0, color="#9A9A9A", linewidth=0.55, zorder=0)
    ax.tick_params(axis="x", labelsize=6.8, length=0, pad=1)
    ax.tick_params(axis="y", labelsize=7, width=0.55, length=2.5, pad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.set_xlabel("")
    ax.set_ylabel("Per-gene Pearson r" if show_ylabel else "", fontsize=7.5)


def add_distribution(ax: plt.Axes, data: list[np.ndarray], metric_key: str) -> None:
    positions = np.arange(1, len(RUNS) + 1)
    violins = ax.violinplot(
        data,
        positions=positions,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, run in zip(violins["bodies"], RUNS):
        body.set_facecolor(PALETTE[run["mode"]])
        body.set_edgecolor(EDGE[run["mode"]])
        body.set_alpha(0.82)
        body.set_linewidth(0.65)

    ax.boxplot(
        data,
        positions=positions,
        widths=0.16,
        patch_artist=True,
        showfliers=False,
        showcaps=False,
        whis=(5, 95),
        boxprops={
            "facecolor": "#FFFFFF",
            "edgecolor": "#2F2F2F",
            "linewidth": 0.65,
            "alpha": 0.82,
        },
        medianprops={"color": "#111111", "linewidth": 0.8},
        whiskerprops={"color": "#2F2F2F", "linewidth": 0.6},
    )

    rng = np.random.default_rng(17)
    for pos, run, vals in zip(positions, RUNS, data):
        if vals.size > 60:
            continue
        else:
            point_size = 6.5
            alpha = 0.34
            jitter = 0.045
        x = pos + rng.normal(0, jitter, vals.size)
        ax.scatter(
            x,
            vals,
            s=point_size,
            color=EDGE[run["mode"]],
            alpha=alpha,
            linewidths=0,
            rasterized=metric_key == "full_spot",
            zorder=2,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([run["label"] for run in RUNS])


def draw(values: dict[str, list[np.ndarray]], outdir: Path, basename: str) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.65,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharey=True)
    for idx, (ax, metric) in enumerate(zip(axes, METRICS)):
        add_distribution(ax, values[metric["key"]], metric["key"])
        style_axes(ax, show_ylabel=idx == 0)
        ax.set_title(metric["title"], loc="left", fontsize=8.2, pad=5)
        ax.text(
            -0.18,
            1.05,
            metric["letter"],
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        ax.text(
            0.98,
            0.96,
            metric["n_label"],
            transform=ax.transAxes,
            fontsize=6.5,
            color="#5A5A5A",
            va="top",
            ha="right",
        )

    handles = [
        Patch(facecolor=PALETTE["within"], edgecolor=EDGE["within"], label="Within section"),
        Patch(facecolor=PALETTE["cross"], edgecolor=EDGE["cross"], label="Cross section"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.03),
        ncol=2,
        frameon=False,
        fontsize=7,
        handlelength=1.2,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.065, right=0.995, top=0.84, bottom=0.19, wspace=0.13)

    pdf_path = outdir / f"{basename}.pdf"
    png_path = outdir / f"{basename}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=360, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    args = parse_args()
    values = load_metric_values(args.result_root)
    pdf_path, png_path = draw(values, args.outdir, args.basename)
    print(f"[saved] {pdf_path}")
    print(f"[saved] {png_path}")


if __name__ == "__main__":
    main()
