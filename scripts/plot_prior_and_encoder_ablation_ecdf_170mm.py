from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]

CELLFM_TABLE_DIR = ROOT / "figures_nature_cellfm_prior_combined_170mm" / "tables"
ENCODER_TABLE_DIR = ROOT / "figures_nature_encoder_ablation_single_multi" / "tables"

OUTDIR = ROOT / "figures_nature_prior_encoder_ablation_ecdf_170mm"
FIG_DIR = OUTDIR / "figures"
TABLE_DIR = OUTDIR / "tables"
PAPER_DIR = (
    Path(r"C:\Users\Administrator\Desktop\人生的第4篇论文")
    / "Figure"
    / "Supplementary_Figure"
    / "Prior_encoder_ablation_ecdf_170mm"
)

CELLFM_ORDER = [
    "Multi + Pretrained",
    "Single + Pretrained",
    "Multi + RandomInit",
    "Single + RandomInit",
]
CELLFM_SHORT = {
    "Multi + Pretrained": "Multi\npretrained",
    "Single + Pretrained": "Single\npretrained",
    "Multi + RandomInit": "Multi\nrandom-init",
    "Single + RandomInit": "Single\nrandom-init",
}
CELLFM_HEAT = {
    "Multi + Pretrained": "Multi\nPre.",
    "Single + Pretrained": "Single\nPre.",
    "Multi + RandomInit": "Multi\nRand.",
    "Single + RandomInit": "Single\nRand.",
}
CELLFM_PALETTE = {
    "Multi + Pretrained": "#E64B35",
    "Single + Pretrained": "#4DBBD5",
    "Multi + RandomInit": "#F39B7F",
    "Single + RandomInit": "#91D1C2",
}

ENCODER_ORDER = [
    ("Multi-slide", "HIPT pathology FM"),
    ("Multi-slide", "ImageNet ResNet50"),
    ("Single-slide", "ImageNet ResNet50"),
    ("Single-slide", "HIPT pathology FM"),
]
ENCODER_CONDITION = {
    ("Multi-slide", "HIPT pathology FM"): "Multi + HIPT",
    ("Multi-slide", "ImageNet ResNet50"): "Multi + ResNet50",
    ("Single-slide", "ImageNet ResNet50"): "Single + ResNet50",
    ("Single-slide", "HIPT pathology FM"): "Single + HIPT",
}
ENCODER_SHORT = {
    "Multi + HIPT": "Multi\nHIPT",
    "Multi + ResNet50": "Multi\nResNet50",
    "Single + ResNet50": "Single\nResNet50",
    "Single + HIPT": "Single\nHIPT",
}
ENCODER_PALETTE = {
    "Multi + HIPT": "#4DBBD5",
    "Multi + ResNet50": "#E64B35",
    "Single + ResNet50": "#F39B7F",
    "Single + HIPT": "#91D1C2",
}

BIN_LABELS = ["Low", "Mid", "High"]
CELLFM_MARKERS = ["KLK3", "AR", "NKX3-1", "AMACR", "ERG", "EPCAM", "KRT8", "KRT18", "KRT5", "TP63"]
ENCODER_MARKERS = ["KLK3", "AR", "NKX3-1", "AMACR", "ERG", "EPCAM", "KRT8", "KRT18", "KRT5", "TP63"]


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", rc={"grid.linewidth": 0.35, "grid.alpha": 0.24})
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def significance_label(pval: float) -> str:
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return "ns"


def annotate_bracket(ax: plt.Axes, x1: int, x2: int, y: float, h: float, text: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="0.18", lw=0.75, clip_on=False)
    ax.text((x1 + x2) / 2, y + h + 0.01, text, ha="center", va="bottom", fontsize=8)


def paired_pvalue_cellfm(gene_df: pd.DataFrame, left: str, right: str) -> float:
    pivot = gene_df[gene_df["model"].isin([left, right])].pivot(index="gene", columns="model", values="pearson")
    pivot = pivot.replace([np.inf, -np.inf], np.nan).dropna()
    try:
        return float(stats.wilcoxon(pivot[left].to_numpy(), pivot[right].to_numpy()).pvalue)
    except Exception:
        return float(stats.ttest_rel(pivot[left].to_numpy(), pivot[right].to_numpy()).pvalue)


def paired_delta_encoder(gene_df: pd.DataFrame, setting: str) -> float:
    subset = gene_df[gene_df["setting"] == setting]
    pivot = subset.pivot(index="gene", columns="encoder", values="pearson")
    pivot = pivot.replace([np.inf, -np.inf], np.nan).dropna()
    return float((pivot["HIPT pathology FM"] - pivot["ImageNet ResNet50"]).mean())


def complementary_ecdf_rows(values: pd.DataFrame, condition_col: str, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for condition, part in values.groupby(condition_col, observed=False):
        corr = part["pearson"].to_numpy(dtype=float)
        corr = corr[np.isfinite(corr)]
        frac = [(corr >= t).mean() for t in thresholds]
        rows.extend(
            {
                condition_col: condition,
                "threshold": float(t),
                "fraction_genes_ge_threshold": float(f),
            }
            for t, f in zip(thresholds, frac)
        )
    return pd.DataFrame(rows)


def load_cellfm() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gene_df = pd.read_csv(require(CELLFM_TABLE_DIR / "cellfm_prior_combined_gene_pearson.csv"))
    summary_df = pd.read_csv(require(CELLFM_TABLE_DIR / "cellfm_prior_combined_summary.csv"))
    marker_df = pd.read_csv(require(CELLFM_TABLE_DIR / "cellfm_prior_combined_marker_pearson.csv"))

    gene_df["model"] = pd.Categorical(gene_df["model"], categories=CELLFM_ORDER, ordered=True)
    summary_df["model"] = pd.Categorical(summary_df["model"], categories=CELLFM_ORDER, ordered=True)
    marker_df["model"] = pd.Categorical(marker_df["model"], categories=CELLFM_ORDER, ordered=True)
    thresholds = np.linspace(-0.2, 1.0, 241)
    ecdf_df = complementary_ecdf_rows(gene_df, "model", thresholds)
    ecdf_df["model"] = pd.Categorical(ecdf_df["model"], categories=CELLFM_ORDER, ordered=True)
    return gene_df, summary_df, marker_df, ecdf_df


def load_encoder() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gene_df = pd.read_csv(require(ENCODER_TABLE_DIR / "pathology_encoder_single_multi_gene_level_pearson.csv"))
    summary_df = pd.read_csv(require(ENCODER_TABLE_DIR / "pathology_encoder_single_multi_summary.csv"))
    marker_df = pd.read_csv(require(ENCODER_TABLE_DIR / "pathology_encoder_single_multi_marker_pearson.csv"))

    gene_df["condition"] = [
        ENCODER_CONDITION[(setting, encoder)] for setting, encoder in zip(gene_df["setting"], gene_df["encoder"])
    ]
    summary_df["condition"] = [
        ENCODER_CONDITION[(setting, encoder)] for setting, encoder in zip(summary_df["setting"], summary_df["encoder"])
    ]
    marker_df["condition"] = [
        ENCODER_CONDITION[(setting, encoder)] for setting, encoder in zip(marker_df["setting"], marker_df["encoder"])
    ]
    encoder_order_names = [ENCODER_CONDITION[item] for item in ENCODER_ORDER]
    gene_df["condition"] = pd.Categorical(gene_df["condition"], categories=encoder_order_names, ordered=True)
    summary_df["condition"] = pd.Categorical(summary_df["condition"], categories=encoder_order_names, ordered=True)
    marker_df["condition"] = pd.Categorical(marker_df["condition"], categories=encoder_order_names, ordered=True)
    thresholds = np.linspace(-0.2, 1.0, 241)
    ecdf_df = complementary_ecdf_rows(gene_df, "condition", thresholds)
    ecdf_df["condition"] = pd.Categorical(ecdf_df["condition"], categories=encoder_order_names, ordered=True)
    return gene_df, summary_df, marker_df, ecdf_df


def plot_violin(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_col: str,
    order: list[str],
    palette: dict[str, str],
    short_labels: dict[str, str],
) -> None:
    sns.violinplot(
        data=data[np.isfinite(data["pearson"])],
        x=x_col,
        y="pearson",
        order=order,
        palette=[palette[k] for k in order],
        inner="box",
        cut=0,
        linewidth=0.62,
        ax=ax,
        hue=x_col,
        legend=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Pearson r")
    ax.set_ylim(-0.52, 1.15)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([short_labels[k] for k in order], rotation=0)
    sns.despine(ax=ax, top=True, right=True)


def plot_ecdf(
    ax: plt.Axes,
    data: pd.DataFrame,
    condition_col: str,
    order: list[str],
    palette: dict[str, str],
    label_map: dict[str, str],
) -> None:
    for condition in order:
        part = data[data[condition_col] == condition]
        ax.plot(
            part["threshold"],
            part["fraction_genes_ge_threshold"],
            color=palette[condition],
            lw=1.25,
            label=label_map[condition].replace("\n", " "),
        )
    ax.set_xlim(-0.2, 1.0)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Pearson r threshold")
    ax.set_ylabel("Fraction of genes")
    ax.legend(frameon=False, loc="upper right", handlelength=1.25, borderaxespad=0.1)
    sns.despine(ax=ax, top=True, right=True)


def plot_strata(
    ax: plt.Axes,
    data: pd.DataFrame,
    condition_col: str,
    order: list[str],
    palette: dict[str, str],
) -> None:
    plot_df = data[data["group"].isin(BIN_LABELS)].copy()
    plot_df["group"] = pd.Categorical(plot_df["group"], categories=BIN_LABELS, ordered=True)
    sns.barplot(
        data=plot_df,
        x="group",
        y="mean_pearson",
        hue=condition_col,
        hue_order=order,
        palette=[palette[k] for k in order],
        ax=ax,
        errorbar=None,
        width=0.86,
    )
    ax.set_xlabel("True-expression stratum")
    ax.set_ylabel("Mean Pearson r")
    ax.set_ylim(0, 0.78)
    ax.get_legend().remove()
    sns.despine(ax=ax, top=True, right=True)


def plot_heatmap(
    ax: plt.Axes,
    marker_df: pd.DataFrame,
    value_col: str,
    column_order: list[str],
    column_labels: list[str],
    marker_order: list[str],
) -> None:
    present = [gene for gene in marker_order if gene in set(marker_df["gene"])]
    pivot = marker_df.pivot(index="gene", columns=value_col, values="pearson").loc[present, column_order]
    pivot.columns = column_labels
    sns.heatmap(
        pivot,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".3f",
        annot_kws={"fontsize": 7.6},
        linewidths=0.36,
        linecolor="white",
        yticklabels=True,
        xticklabels=True,
        cbar_kws={"label": "Pearson r", "shrink": 0.72, "pad": 0.02},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0, pad=1)
    ax.tick_params(axis="y", rotation=0, pad=1)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.0,
        1.10,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
    )


def copy_outputs(paths: list[Path]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, PAPER_DIR / path.name)


def main() -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    cell_gene, cell_summary, cell_marker, cell_ecdf = load_cellfm()
    enc_gene, enc_summary, enc_marker, enc_ecdf = load_encoder()

    cell_ecdf.to_csv(TABLE_DIR / "cellfm_prior_complementary_ecdf.csv", index=False)
    enc_ecdf.to_csv(TABLE_DIR / "pathology_encoder_complementary_ecdf.csv", index=False)

    encoder_order_names = [ENCODER_CONDITION[item] for item in ENCODER_ORDER]
    width_in = 170 / 25.4
    fig = plt.figure(figsize=(width_in, 9.15), constrained_layout=False)
    gs = fig.add_gridspec(
        4,
        2,
        height_ratios=[1.0, 1.32, 1.0, 1.32],
        width_ratios=[1.0, 1.0],
        hspace=0.66,
        wspace=0.36,
    )
    axes = [fig.add_subplot(gs[i, j]) for i in range(4) for j in range(2)]
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h = axes

    plot_violin(ax_a, cell_gene, "model", CELLFM_ORDER, CELLFM_PALETTE, CELLFM_SHORT)
    ax_a.set_title("CellFM prior: per-gene distribution")
    p_multi = paired_pvalue_cellfm(cell_gene, "Multi + Pretrained", "Multi + RandomInit")
    p_single = paired_pvalue_cellfm(cell_gene, "Single + Pretrained", "Single + RandomInit")
    annotate_bracket(ax_a, 0, 2, 1.03, 0.035, significance_label(p_multi))
    annotate_bracket(ax_a, 1, 3, 1.10, 0.035, significance_label(p_single))

    plot_ecdf(ax_b, cell_ecdf, "model", CELLFM_ORDER, CELLFM_PALETTE, CELLFM_SHORT)
    ax_b.set_title("CellFM prior: complementary ECDF")

    plot_strata(ax_c, cell_summary, "model", CELLFM_ORDER, CELLFM_PALETTE)
    ax_c.set_title("CellFM prior: true-expression strata")
    handles = [mpl.patches.Patch(facecolor=CELLFM_PALETTE[k], edgecolor="0.25", lw=0.4, label=CELLFM_SHORT[k].replace("\n", " ")) for k in CELLFM_ORDER]
    ax_c.legend(handles=handles, frameon=False, loc="upper left", ncol=1, borderaxespad=0.1)

    plot_heatmap(
        ax_d,
        cell_marker,
        "model",
        CELLFM_ORDER,
        [CELLFM_HEAT[k] for k in CELLFM_ORDER],
        CELLFM_MARKERS,
    )
    ax_d.set_title("CellFM prior: marker genes")

    plot_violin(ax_e, enc_gene, "condition", encoder_order_names, ENCODER_PALETTE, ENCODER_SHORT)
    ax_e.set_title("Pathology encoder: per-gene distribution")
    single_delta = paired_delta_encoder(enc_gene, "Single-slide")
    multi_delta = paired_delta_encoder(enc_gene, "Multi-slide")
    ax_e.text(
        0.02,
        0.96,
        f"HIPT - ResNet50: multi {multi_delta:+.003f}; single {single_delta:+.003f}",
        transform=ax_e.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
    )

    plot_ecdf(ax_f, enc_ecdf, "condition", encoder_order_names, ENCODER_PALETTE, ENCODER_SHORT)
    ax_f.set_title("Pathology encoder: complementary ECDF")

    plot_strata(ax_g, enc_summary, "condition", encoder_order_names, ENCODER_PALETTE)
    ax_g.set_title("Pathology encoder: true-expression strata")
    handles = [
        mpl.patches.Patch(facecolor=ENCODER_PALETTE[k], edgecolor="0.25", lw=0.4, label=ENCODER_SHORT[k].replace("\n", " "))
        for k in encoder_order_names
    ]
    ax_g.legend(handles=handles, frameon=False, loc="upper left", ncol=1, borderaxespad=0.1)

    plot_heatmap(
        ax_h,
        enc_marker,
        "condition",
        encoder_order_names,
        ["Multi\nHIPT", "Multi\nRN50", "Single\nRN50", "Single\nHIPT"],
        ENCODER_MARKERS,
    )
    ax_h.set_title("Pathology encoder: marker genes")

    for label, ax in zip("abcdefgh", axes):
        add_panel_label(ax, label)

    fig.subplots_adjust(left=0.085, right=0.965, bottom=0.055, top=0.965)

    outputs = []
    stem = "prior_encoder_ablation_ecdf_170mm"
    for ext in ["pdf", "svg", "png"]:
        path = FIG_DIR / f"{stem}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=600)
        else:
            fig.savefig(path)
        outputs.append(path)
    plt.close(fig)
    copy_outputs(outputs)

    combined_summary = pd.concat(
        [
            cell_summary[cell_summary["group"] == "All"].assign(ablation="CellFM prior"),
            enc_summary[enc_summary["group"] == "All"].assign(ablation="Pathology encoder"),
        ],
        ignore_index=True,
    )
    combined_summary.to_csv(TABLE_DIR / "prior_encoder_ablation_all_gene_summary.csv", index=False)
    shutil.copy2(TABLE_DIR / "prior_encoder_ablation_all_gene_summary.csv", PAPER_DIR / "prior_encoder_ablation_all_gene_summary.csv")

    print(f"Saved combined figure to {FIG_DIR}")
    print(f"Copied paper-ready files to {PAPER_DIR}")
    print(combined_summary.to_string(index=False))


if __name__ == "__main__":
    main()
