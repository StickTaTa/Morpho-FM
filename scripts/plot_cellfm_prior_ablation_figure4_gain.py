from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "figures_nature_cellfm_ablation"
TABLE_DIR = SOURCE / "tables"
OUTDIR = ROOT / "figures_nature_cellfm_ablation_revision"
PAPER_FIG_DIR = Path(r"C:\Users\Administrator\Desktop\人生的第4篇论文\Figure\Figure_4")

MODEL_ORDER = [
    "Multi + Pretrained",
    "Multi + RandomInit",
    "Single + Pretrained",
    "Single + RandomInit",
]
SETTING_ORDER = ["Single-slide", "Multi-slide"]
INIT_ORDER = ["Pretrained CellFM", "Random initialized CellFM"]

MODEL_LABELS = {
    "Single + Pretrained": "Single\nPT",
    "Single + RandomInit": "Single\nRI",
    "Multi + Pretrained": "Multi\nPT",
    "Multi + RandomInit": "Multi\nRI",
}
INIT_LABELS = {
    "Pretrained CellFM": "Pretrained",
    "Random initialized CellFM": "Random init",
}

PALETTE = {
    "Pretrained CellFM": "#4DBBD5",
    "Random initialized CellFM": "#E64B35",
}

MARKER_GENES = ["KLK3", "AR", "NKX3-1", "AMACR", "ERG", "EPCAM", "KRT8", "KRT18", "KRT5", "TP63"]
BIN_ORDER = ["Low", "Mid", "High"]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
        }
    )
    sns.set_theme(
        style="whitegrid",
        rc={
            "grid.linewidth": 0.55,
            "grid.alpha": 0.25,
            "axes.facecolor": "white",
        },
    )


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gene = pd.read_csv(TABLE_DIR / "cellfm_prior_gene_level_pearson.csv")
    summary = pd.read_csv(TABLE_DIR / "cellfm_prior_summary_by_group.csv")
    delta = pd.read_csv(TABLE_DIR / "cellfm_prior_delta_pretrained_minus_randominit.csv")
    marker = pd.read_csv(TABLE_DIR / "cellfm_prior_marker_pearson.csv")
    return gene, summary, delta, marker


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.105,
        1.065,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def add_design_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["setting"] = out["model"].map(
        {
            "Single + Pretrained": "Single-slide",
            "Single + RandomInit": "Single-slide",
            "Multi + Pretrained": "Multi-slide",
            "Multi + RandomInit": "Multi-slide",
        }
    )
    out["initialization"] = out["model"].map(
        {
            "Single + Pretrained": "Pretrained CellFM",
            "Multi + Pretrained": "Pretrained CellFM",
            "Single + RandomInit": "Random initialized CellFM",
            "Multi + RandomInit": "Random initialized CellFM",
        }
    )
    out["setting"] = pd.Categorical(out["setting"], categories=SETTING_ORDER, ordered=True)
    out["initialization"] = pd.Categorical(out["initialization"], categories=INIT_ORDER, ordered=True)
    return out


def draw_panel_a(ax: plt.Axes, gene: pd.DataFrame, delta: pd.DataFrame) -> None:
    plot = gene[np.isfinite(gene["pearson"])].copy()
    plot = add_design_columns(plot)

    sns.violinplot(
        data=plot,
        x="setting",
        y="pearson",
        hue="initialization",
        order=SETTING_ORDER,
        hue_order=INIT_ORDER,
        palette=[PALETTE[i] for i in INIT_ORDER],
        inner="box",
        cut=0,
        linewidth=0.85,
        ax=ax,
        legend=True,
    )

    means = (
        plot.groupby(["setting", "initialization"], observed=False)["pearson"]
        .mean()
        .reindex(pd.MultiIndex.from_product([SETTING_ORDER, INIT_ORDER], names=["setting", "initialization"]))
    )
    offsets = {"Pretrained CellFM": -0.2, "Random initialized CellFM": 0.2}
    for setting_idx, setting in enumerate(SETTING_ORDER):
        for initialization in INIT_ORDER:
            ax.scatter(
                setting_idx + offsets[initialization],
                means.loc[(setting, initialization)],
                marker="D",
                s=16,
                c="black",
                zorder=5,
            )

    delta_all = delta[delta["group"] == "All"].set_index("protocol")
    annotations = [
        (0, float(delta_all.loc["Single", "delta_mean_pearson"])),
        (1, float(delta_all.loc["Multi", "delta_mean_pearson"])),
    ]
    for setting_idx, value in annotations:
        left = setting_idx - 0.2
        right = setting_idx + 0.2
        y = 1.015
        ax.plot([left, left, right, right], [y, y + 0.035, y + 0.035, y], color="black", lw=0.65)
        ax.text((left + right) / 2, y + 0.04, rf"$\Delta$mean={value:+.4f}", ha="center", va="bottom", fontsize=7.0)

    ax.axhline(0, color="black", linewidth=0.55, alpha=0.5)
    ax.set_title("Per-gene Pearson distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Per-gene Pearson r")
    ax.set_ylim(-1.0, 1.15)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[:2],
        [INIT_LABELS[label] for label in labels[:2]],
        frameon=False,
        loc="lower left",
        ncol=1,
        borderaxespad=0.2,
    )
    sns.despine(ax=ax, top=True, right=True)
    panel_label(ax, "a")


def draw_panel_b(fig: plt.Figure, subgrid, summary: pd.DataFrame) -> None:
    plot = summary[summary["group"].isin(BIN_ORDER)].copy()
    plot = add_design_columns(plot)
    plot["group"] = pd.Categorical(plot["group"], categories=BIN_ORDER, ordered=True)
    axes = [fig.add_subplot(subgrid[0, 0]), fig.add_subplot(subgrid[0, 1])]
    axes[1].sharey(axes[0])
    for ax, setting in zip(axes, SETTING_ORDER):
        sns.barplot(
            data=plot[plot["setting"] == setting],
            x="group",
            y="mean_pearson",
            hue="initialization",
            hue_order=INIT_ORDER,
            palette=[PALETTE[i] for i in INIT_ORDER],
            ax=ax,
        )
        ax.set_title(setting)
        ax.set_xlabel("True-expression bin")
        ax.set_ylabel("Mean Pearson r" if ax is axes[0] else "")
        ax.set_ylim(0, 0.79)
        ax.get_legend().remove()
        sns.despine(ax=ax, top=True, right=True)
    axes[1].tick_params(labelleft=False)
    axes[1].spines["left"].set_visible(False)
    panel_label(axes[0], "b")


def draw_panel_c(ax: plt.Axes, marker: pd.DataFrame) -> None:
    column_order = [
        "Single + Pretrained",
        "Single + RandomInit",
        "Multi + Pretrained",
        "Multi + RandomInit",
    ]
    pivot = marker.pivot(index="gene", columns="model", values="pearson").loc[MARKER_GENES, column_order]
    pivot.columns = [MODEL_LABELS[m] for m in column_order]
    sns.heatmap(
        pivot,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".3f",
        linewidths=0.45,
        cbar_kws={"label": "Pearson r"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Fixed prostate marker panel")
    ax.tick_params(axis="x", rotation=0)
    panel_label(ax, "c")


def main() -> None:
    configure_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    gene, summary, delta, marker = load_tables()
    fig = plt.figure(figsize=(11.7, 3.65), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.85, 2.25, 1.9], wspace=0.36)
    draw_panel_a(fig.add_subplot(gs[0, 0]), gene, delta)
    draw_panel_b(fig, gs[0, 1].subgridspec(1, 2, wspace=0.16), summary)
    draw_panel_c(fig.add_subplot(gs[0, 2]), marker)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.82, bottom=0.20)

    for suffix in ["png", "pdf", "svg", "tif"]:
        out = OUTDIR / f"Figure_4.{suffix}"
        fig.savefig(out, bbox_inches="tight")
        shutil.copy2(out, PAPER_FIG_DIR / f"Figure_4.{suffix}")
    plt.close(fig)

    summary_copy = OUTDIR / "figure4_source_summary_by_group.csv"
    delta_copy = OUTDIR / "figure4_source_delta.csv"
    summary.to_csv(summary_copy, index=False)
    delta.to_csv(delta_copy, index=False)
    shutil.copy2(summary_copy, PAPER_FIG_DIR / summary_copy.name)
    shutil.copy2(delta_copy, PAPER_FIG_DIR / delta_copy.name)
    print(f"Saved revised Figure 4 to {OUTDIR}")
    print(f"Copied revised Figure 4 to {PAPER_FIG_DIR}")


if __name__ == "__main__":
    main()
