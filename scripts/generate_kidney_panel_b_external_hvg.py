from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


ROOT = Path(os.environ.get("MORPHO_VC_ROOT", Path(__file__).resolve().parents[1])).resolve()
RESULT_ROOT = ROOT / "results" / "other_kidney"
OUTDIR_DEFAULT = ROOT / "figures_nature_kidney" / "panel_b_generalization"
NEUTRAL_GRAY = "#4D4D4D"

SERIES = [
    {
        "key": "INT21_holdout",
        "label": "INT21 benchmark",
        "color": NEUTRAL_GRAY,
        "result_dir": ROOT / "results" / "kidney_hest" / "morpho" / "multi_superres" / "INT21_holdout",
    },
    {
        "key": "GSM6415706_GLMF1",
        "label": "GLMF1",
        "color": "#3C5488",
        "result_dir": RESULT_ROOT / "GSM6415706_GLMF1" / "multi",
    },
    {
        "key": "GSM6415706_GLMF2",
        "label": "GLMF2",
        "color": "#E64B35",
        "result_dir": RESULT_ROOT / "GSM6415706_GLMF2" / "multi",
    },
]

SERIES_KEYS = [item["key"] for item in SERIES]
SERIES_LABELS = {item["key"]: item["label"] for item in SERIES}
SERIES_COLORS = {item["key"]: item["color"] for item in SERIES}
SERIES_RESULT_DIRS = {item["key"]: item["result_dir"] for item in SERIES}

MM_TO_INCH = 1.0 / 25.4
FIG_WIDTH_MM = 85.0
FIG_HEIGHT_MM_SINGLE = 70.0
FIG_HEIGHT_MM_STACKED = 120.0
FIG_HEIGHT_MM_INSET = 72.0
FIG_HEIGHT_MM_STRIP = 92.0
BOTTOM_PANEL_WIDTH_MM = 42.82
BOTTOM_PANEL_HEIGHT_MM = 31.803
BOTTOM_PANEL_BOX_ASPECT = BOTTOM_PANEL_HEIGHT_MM / BOTTOM_PANEL_WIDTH_MM


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    }
)
sns.set_theme(style="whitegrid", rc={"grid.linewidth": 0.6, "grid.alpha": 0.20, "axes.facecolor": "white"})


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(axis="x", labelsize=8, pad=1)
    ax.tick_params(axis="y", labelsize=8, pad=1)
    sns.despine(ax=ax, top=True, right=True)


def req(path: Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def log1p_nonneg(x: np.ndarray) -> None:
    np.clip(x, 0, None, out=x)
    np.log1p(x, out=x)


def rank_var(mat: np.ndarray) -> np.ndarray:
    x = np.asarray(mat, dtype=np.float32).copy()
    log1p_nonneg(x)
    return np.argsort(np.var(x, axis=0))[::-1]


def ksize(k: int, kmax: int) -> float:
    frac = min(max(k / max(kmax, 1), 0.0), 1.0)
    return 10 + (70 - 10) * (frac ** 0.7)


def load_slide_result(slide_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result_dir = SERIES_RESULT_DIRS[slide_id]
    pred = np.load(req(result_dir / "pred_bag.npy"))
    true = np.load(req(result_dir / "true_bag.npy"))
    pear = np.load(req(result_dir / "gene_corr_all.npy"))
    return pred, true, pear


def compute_hvg_rows(slide_id: str) -> pd.DataFrame:
    print(f"[slide] loading {slide_id}")
    pred, true, pear = load_slide_result(slide_id)
    tr = rank_var(true)
    pr = rank_var(pred)
    n_genes = int(pred.shape[1])
    kmax = min(2000, n_genes)
    ks = np.array([kmax], dtype=int) if kmax < 50 else np.arange(50, kmax + 1, 50, dtype=int)

    rows = []
    for k in ks:
        tp = pr[:k]
        tt = tr[:k]
        ov = np.intersect1d(tt, tp).size
        rows.append(
            {
                "slide_id": slide_id,
                "slide_label": SERIES_LABELS[slide_id],
                "K": int(k),
                "Recall@K": float(ov / float(k)),
                "MeanPearson_on_HVGpred": float(np.nanmean(pear[tp])),
                "Overlap": int(ov),
                "Kmax": int(kmax),
            }
        )

    del pred, true, pear, tr, pr
    gc.collect()
    return pd.DataFrame(rows)


def compute_threshold_rows(slide_id: str) -> pd.DataFrame:
    result_dir = SERIES_RESULT_DIRS[slide_id]
    pear = np.load(req(result_dir / "gene_corr_all.npy"))
    pear = np.asarray(pear, dtype=np.float32)
    pear = pear[np.isfinite(pear)]
    thr = np.linspace(-0.2, 1.0, 251)
    rows = [
        {
            "slide_id": slide_id,
            "slide_label": SERIES_LABELS[slide_id],
            "Threshold": float(t),
            "Fraction": float((pear >= t).mean()),
        }
        for t in thr
    ]
    return pd.DataFrame(rows)


def compute_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    hvg_df = pd.concat([compute_hvg_rows(slide_id) for slide_id in SERIES_KEYS], ignore_index=True)
    thr_df = pd.concat([compute_threshold_rows(slide_id) for slide_id in SERIES_KEYS], ignore_index=True)
    return hvg_df, thr_df


def plot_threshold(ax: plt.Axes, thr_df: pd.DataFrame, show_legend: bool = True) -> None:
    for slide_id in SERIES_KEYS:
        sub = thr_df[thr_df["slide_id"] == slide_id].sort_values("Threshold")
        ax.plot(
            sub["Threshold"],
            sub["Fraction"],
            label=SERIES_LABELS[slide_id],
            color=SERIES_COLORS[slide_id],
            linewidth=1.4,
        )
    ax.set_xlabel("Threshold (Pearson r)")
    ax.set_ylabel("Fraction of genes")
    ax.set_xlim(-0.2, 1.0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linewidth=0.6, alpha=0.18)
    if show_legend:
        ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    style_axis(ax)


def plot_hvg(ax: plt.Axes, hvg_df: pd.DataFrame, show_legends: bool = True) -> None:
    for slide_id in SERIES_KEYS:
        sub = hvg_df[hvg_df["slide_id"] == slide_id].sort_values("K")
        kmax = int(sub["Kmax"].iloc[0])
        color = SERIES_COLORS[slide_id]
        ax.plot(
            sub["Recall@K"],
            sub["MeanPearson_on_HVGpred"],
            color=color,
            linewidth=1.6,
            alpha=0.85,
        )
        ax.scatter(
            sub["Recall@K"],
            sub["MeanPearson_on_HVGpred"],
            s=[ksize(int(k), kmax) for k in sub["K"]],
            color=color,
            alpha=0.35,
            edgecolor="none",
        )
        highlight_ks = [50, 200, 500, 1000, 2000]
        subh = sub[sub["K"].isin([k for k in highlight_ks if k <= kmax])]
        ax.scatter(
            subh["Recall@K"],
            subh["MeanPearson_on_HVGpred"],
            s=[ksize(int(k), kmax) * 1.05 for k in subh["K"]],
            color=color,
            alpha=0.95,
            edgecolor="black",
            linewidth=0.35,
        )

    ax.set_xlabel("HVG overlap (Recall@K)")
    ax.set_ylabel("Mean Pearson on HVG_pred(topK)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.1, 1.0)
    ax.grid(True, linewidth=0.6, alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legends:
        line_handles = [
            Line2D([0], [0], color=SERIES_COLORS[s], lw=2.0, label=SERIES_LABELS[s])
            for s in SERIES_KEYS
        ]
        leg1 = ax.legend(handles=line_handles, frameon=False, loc="upper left")

        max_k = int(hvg_df["Kmax"].max())
        size_keys = [k for k in [50, 200, 500, 1000, 2000] if k <= max_k]
        size_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor="#9A9A9A",
                markeredgecolor=NEUTRAL_GRAY,
                markeredgewidth=0.35,
                markersize=np.sqrt(ksize(k, max_k)),
                label=f"K={k}",
            )
            for k in size_keys
        ]
        if size_handles:
            ax.legend(handles=size_handles, frameon=False, loc="lower right", handletextpad=0.6, labelspacing=0.25)
            ax.add_artist(leg1)
    style_axis(ax)


def plot_threshold_inset(ax: plt.Axes, thr_df: pd.DataFrame) -> None:
    for slide_id in SERIES_KEYS:
        sub = thr_df[thr_df["slide_id"] == slide_id].sort_values("Threshold")
        ax.plot(
            sub["Threshold"],
            sub["Fraction"],
            color=SERIES_COLORS[slide_id],
            linewidth=1.1,
        )
    ax.set_xlim(-0.2, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(labelsize=6.5, pad=1)
    ax.set_title("Fraction >= threshold", fontsize=6.8, pad=2)
    ax.grid(True, linewidth=0.45, alpha=0.16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    style_axis(ax)


def save_fig(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)


def remove_old_fullk_outputs(outdir: Path) -> None:
    for path in [
        outdir / "panel_b_external_hvg_recovery_fullK.csv",
        outdir / "panel_b_external_hvg_recovery_fullK.pdf",
        outdir / "panel_b_external_hvg_recovery_fullK.png",
    ]:
        if path.exists():
            path.unlink()


def plot_all(hvg_df: pd.DataFrame, thr_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    remove_old_fullk_outputs(outdir)

    fig_thr, ax_thr = plt.subplots(figsize=(FIG_WIDTH_MM * MM_TO_INCH, FIG_HEIGHT_MM_SINGLE * MM_TO_INCH))
    plot_threshold(ax_thr, thr_df)
    fig_thr.tight_layout(pad=0.5)
    save_fig(fig_thr, outdir / "panel_b_generalization_fraction_threshold")

    fig_hvg, ax_hvg = plt.subplots(figsize=(FIG_WIDTH_MM * MM_TO_INCH, FIG_HEIGHT_MM_SINGLE * MM_TO_INCH))
    plot_hvg(ax_hvg, hvg_df)
    fig_hvg.tight_layout(pad=0.5)
    save_fig(fig_hvg, outdir / "panel_b_generalization_hvg_denseK")

    fig_stack, axes = plt.subplots(
        2,
        1,
        figsize=(FIG_WIDTH_MM * MM_TO_INCH, FIG_HEIGHT_MM_STACKED * MM_TO_INCH),
        sharex=False,
    )
    plot_threshold(axes[0], thr_df)
    plot_hvg(axes[1], hvg_df)
    axes[0].set_title("Fraction of genes with Pearson r >= threshold", pad=3)
    axes[1].set_title("HVG recovery vs prediction quality", pad=3)
    fig_stack.tight_layout(pad=0.5, h_pad=1.0)
    save_fig(fig_stack, outdir / "panel_b_generalization_threshold_plus_hvg")

    fig_inset, ax_main = plt.subplots(
        figsize=(FIG_WIDTH_MM * MM_TO_INCH, FIG_HEIGHT_MM_INSET * MM_TO_INCH)
    )
    plot_hvg(ax_main, hvg_df)
    ax_main.set_title("HVG recovery vs prediction quality", pad=3)
    ax_in = inset_axes(
        ax_main,
        width="42%",
        height="42%",
        loc="upper right",
        borderpad=0.9,
    )
    plot_threshold_inset(ax_in, thr_df)
    fig_inset.subplots_adjust(left=0.16, right=0.98, top=0.95, bottom=0.12)
    save_fig(fig_inset, outdir / "panel_b_generalization_hvg_with_threshold_inset")

    fig_strip = plt.figure(
        figsize=(FIG_WIDTH_MM * MM_TO_INCH, FIG_HEIGHT_MM_STRIP * MM_TO_INCH)
    )
    gs = fig_strip.add_gridspec(
        2,
        1,
        height_ratios=[0.55, 1.45],
        hspace=0.14,
    )
    ax_top = fig_strip.add_subplot(gs[0, 0])
    ax_bottom = fig_strip.add_subplot(gs[1, 0])
    plot_threshold(ax_top, thr_df, show_legend=False)
    plot_hvg(ax_bottom, hvg_df, show_legends=True)
    ax_bottom.set_box_aspect(BOTTOM_PANEL_BOX_ASPECT)
    ax_top.set_title("Fraction of genes with Pearson r >= threshold", pad=3)
    ax_top.set_xlabel("")
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_bottom.set_title("HVG recovery vs prediction quality", pad=3)
    fig_strip.subplots_adjust(left=0.16, right=0.98, top=0.96, bottom=0.09, hspace=0.14)
    save_fig(fig_strip, outdir / "panel_b_generalization_threshold_strip_plus_hvg")

    thr_df.to_csv(outdir / "panel_b_generalization_fraction_threshold.csv", index=False)
    hvg_df.to_csv(outdir / "panel_b_generalization_hvg_denseK.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate external-slide panel b using notebook-style threshold and HVG plots.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=OUTDIR_DEFAULT,
        help="Output directory for panel b.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hvg_df, thr_df = compute_data()
    plot_all(hvg_df, thr_df, args.outdir)
    print("[done] panel b generated:")
    for path in sorted(args.outdir.iterdir()):
        print(f"  - {path}")


if __name__ == "__main__":
    main()
