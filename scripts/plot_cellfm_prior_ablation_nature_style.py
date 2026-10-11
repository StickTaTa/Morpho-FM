from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


ROOT = Path(os.environ.get("MORPHO_VC_ROOT", Path.cwd())).expanduser().resolve()
if not (ROOT / "results").exists() and (ROOT.parent / "results").exists():
    ROOT = ROOT.parent

TEST_ID = "INT28"

RESULT_DIRS = {
    "Multi + Pretrained": ROOT / "results" / "st_superres" / "cellfm_INT28",
    "Multi + RandomInit": ROOT / "results" / "st_superres" / "cellfm_randominit_ablation" / "multi_seed42_INT28",
    "Single + Pretrained": ROOT / "results" / "prostate_hest" / "morpho" / "single_superres" / "INT25_to_INT28",
    "Single + RandomInit": ROOT / "results" / "st_superres" / "cellfm_randominit_ablation" / "single_seed42_INT28",
}

ORDER = [
    "Multi + Pretrained",
    "Multi + RandomInit",
    "Single + Pretrained",
    "Single + RandomInit",
]

COMBINED_ORDER = [
    "Multi + Pretrained",
    "Single + Pretrained",
    "Multi + RandomInit",
    "Single + RandomInit",
]
COMBINED_LABELS = {
    "Multi + Pretrained": "Multi pretrained",
    "Single + Pretrained": "Single pretrained",
    "Multi + RandomInit": "Multi random-init",
    "Single + RandomInit": "Single random-init",
}

PALETTE = {
    "Multi + Pretrained": "#E64B35",
    "Multi + RandomInit": "#F39B7F",
    "Single + Pretrained": "#4DBBD5",
    "Single + RandomInit": "#91D1C2",
}

MARKER_GENES = [
    "KLK3",
    "AR",
    "NKX3-1",
    "AMACR",
    "ERG",
    "EPCAM",
    "KRT8",
    "KRT18",
    "KRT5",
    "TP63",
]

BIN_LABELS = ["Low", "Mid", "High"]
Q_LOW, Q_HIGH = 1 / 3, 2 / 3

OUTDIR = ROOT / "figures_nature_cellfm_ablation"
TABLE_DIR = OUTDIR / "tables"
FIG_DIR = OUTDIR / "figures"
PAPER_FIG_DIR = Path(r"C:\Users\Administrator\Desktop\人生的第4篇论文\Figure\Figure_4")


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
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
            "grid.linewidth": 0.6,
            "grid.alpha": 0.25,
            "axes.facecolor": "white",
        },
    )


def save_fig(fig: plt.Figure, outdir: Path, name: str, tight: bool = True) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    bbox = "tight" if tight else None
    fig.savefig(outdir / f"{name}.pdf", bbox_inches=bbox)
    fig.savefig(outdir / f"{name}.svg", bbox_inches=bbox)
    fig.savefig(outdir / f"{name}.png", bbox_inches=bbox)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_pair(result_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    pred = np.load(require_file(result_dir / "pred_bag.npy"))
    true = np.load(require_file(result_dir / "true_bag.npy"))
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch in {result_dir}: pred={pred.shape}, true={true.shape}")
    return pred, true


def load_gene_list(n_genes: int) -> list[str]:
    candidates = [
        ROOT / "checkpoints" / "st_superres" / "cellfm_forward_sum" / "common_genes.json",
        ROOT / "checkpoints" / "st_superres" / "cellfm_randominit_ablation" / "multi_seed42" / "common_genes.json",
        ROOT / "checkpoints" / "st_superres" / "cellfm_randominit_ablation" / "single_seed42" / "common_genes.json",
        ROOT / "checkpoints" / "st_superres" / "cellfm_single_INT25_to_INT28" / "common_genes.json",
        ROOT / "data" / "spatial_data" / "common_genes.txt",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".json":
            genes = json.loads(path.read_text(encoding="utf-8"))
        else:
            genes = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(genes) == n_genes:
            return [str(g) for g in genes]
    raise ValueError(f"Cannot find a gene list matching n_genes={n_genes}")


def pearson_corr_per_gene(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    p = pred - pred.mean(axis=0, keepdims=True)
    t = true - true.mean(axis=0, keepdims=True)
    num = np.sum(p * t, axis=0)
    den = np.sqrt(np.sum(p * p, axis=0) * np.sum(t * t, axis=0)) + 1e-12
    corr = num / den
    corr[~np.isfinite(corr)] = np.nan
    return corr


def significance_label(pval: float) -> str:
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return "ns"


def annotate_pval(ax: plt.Axes, idx1: int, idx2: int, y: float, h: float, pval: float) -> None:
    ax.plot([idx1, idx1, idx2, idx2], [y, y + h, y + h, y], lw=1.0, c="black")
    ax.text(
        (idx1 + idx2) * 0.5,
        y + h + 0.02,
        f"({significance_label(float(pval))})",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )


def paired_pvalue(df: pd.DataFrame, left: str, right: str) -> float:
    pivot = df[df["model"].isin([left, right])].pivot(index="gene", columns="model", values="pearson")
    pivot = pivot.replace([np.inf, -np.inf], np.nan).dropna()
    if pivot.empty:
        return np.nan
    try:
        _, pval = stats.wilcoxon(pivot[left].values, pivot[right].values)
    except Exception:
        _, pval = stats.ttest_rel(pivot[left].values, pivot[right].values)
    return float(pval)


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    pairs: dict[str, dict[str, np.ndarray]] = {}
    ref_true = None
    n_genes = None

    for model_name, result_dir in RESULT_DIRS.items():
        pred, true = load_pair(result_dir)
        if n_genes is None:
            n_genes = pred.shape[1]
        elif pred.shape[1] != n_genes:
            raise ValueError(f"Gene count mismatch for {model_name}: {pred.shape[1]} vs {n_genes}")

        if ref_true is None:
            ref_true = true
        else:
            if true.shape != ref_true.shape:
                raise ValueError(f"True matrix shape mismatch for {model_name}: {true.shape} vs {ref_true.shape}")
            max_abs = float(np.max(np.abs(true - ref_true)))
            if max_abs > 1e-5:
                raise ValueError(f"True matrix differs for {model_name}; max abs difference={max_abs}")
        pairs[model_name] = {"pred": pred, "true": true}

    if ref_true is None or n_genes is None:
        raise RuntimeError("No result pairs loaded.")

    genes = load_gene_list(n_genes)
    true_gene_intensity = np.log1p(np.clip(ref_true, 0, None)).mean(axis=0)
    q1, q2 = np.quantile(true_gene_intensity, [Q_LOW, Q_HIGH])
    bin_idx = np.zeros_like(true_gene_intensity, dtype=np.int64)
    bin_idx[true_gene_intensity > q1] = 1
    bin_idx[true_gene_intensity > q2] = 2
    bin_name = np.array(BIN_LABELS, dtype=object)[bin_idx]

    gene_rows = []
    summary_rows = []
    all_corrs = {}

    for model_name in ORDER:
        corr = pearson_corr_per_gene(pairs[model_name]["pred"], ref_true)
        all_corrs[model_name] = corr

        for gi, gene in enumerate(genes):
            gene_rows.append(
                {
                    "model": model_name,
                    "gene": gene,
                    "pearson": float(corr[gi]),
                    "true_expr_intensity": float(true_gene_intensity[gi]),
                    "expr_bin": str(bin_name[gi]),
                }
            )

        summary_rows.append(
            {
                "model": model_name,
                "group": "All",
                "n_genes": int(np.isfinite(corr).sum()),
                "mean_pearson": float(np.nanmean(corr)),
                "median_pearson": float(np.nanmedian(corr)),
            }
        )
        for group in BIN_LABELS:
            mask = bin_name == group
            values = corr[mask]
            summary_rows.append(
                {
                    "model": model_name,
                    "group": group,
                    "n_genes": int(np.isfinite(values).sum()),
                    "mean_pearson": float(np.nanmean(values)),
                    "median_pearson": float(np.nanmedian(values)),
                }
            )

    gene_df = pd.DataFrame(gene_rows)
    summary_df = pd.DataFrame(summary_rows)

    delta_rows = []
    for protocol in ["Multi", "Single"]:
        pretrained = f"{protocol} + Pretrained"
        randominit = f"{protocol} + RandomInit"
        for group in ["All"] + BIN_LABELS:
            pre = summary_df[(summary_df.model == pretrained) & (summary_df.group == group)]["mean_pearson"].iloc[0]
            rnd = summary_df[(summary_df.model == randominit) & (summary_df.group == group)]["mean_pearson"].iloc[0]
            delta_rows.append(
                {
                    "protocol": protocol,
                    "group": group,
                    "delta_mean_pearson": float(pre - rnd),
                    "pretrained_mean": float(pre),
                    "randominit_mean": float(rnd),
                }
            )
    delta_df = pd.DataFrame(delta_rows)

    marker_keep = [g for g in MARKER_GENES if g in genes]
    marker_rows = []
    for model_name in ORDER:
        corr = all_corrs[model_name]
        for gene in marker_keep:
            gi = genes.index(gene)
            marker_rows.append({"model": model_name, "gene": gene, "pearson": float(corr[gi])})
    marker_df = pd.DataFrame(marker_rows)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    gene_df.to_csv(TABLE_DIR / "cellfm_prior_gene_level_pearson.csv", index=False, encoding="utf-8")
    summary_df.to_csv(TABLE_DIR / "cellfm_prior_summary_by_group.csv", index=False, encoding="utf-8")
    delta_df.to_csv(TABLE_DIR / "cellfm_prior_delta_pretrained_minus_randominit.csv", index=False, encoding="utf-8")
    marker_df.to_csv(TABLE_DIR / "cellfm_prior_marker_pearson.csv", index=False, encoding="utf-8")

    return gene_df, summary_df, delta_df, marker_df, all_corrs


def plot_a1_violin(gene_df: pd.DataFrame) -> None:
    plot_df = gene_df[np.isfinite(gene_df["pearson"])].copy()
    plot_df["model"] = pd.Categorical(plot_df["model"], categories=ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(6.55, 3.8), dpi=300)
    sns.violinplot(
        data=plot_df,
        x="model",
        y="pearson",
        order=ORDER,
        palette=[PALETTE[m] for m in ORDER],
        inner="box",
        cut=0,
        linewidth=0.9,
        ax=ax,
        hue="model",
        legend=False,
    )
    ax.set_title(f"CellFM prior ablation on {TEST_ID}: per-gene Pearson")
    ax.set_xlabel("")
    ax.set_ylabel("Pearson r")
    ax.set_ylim(-0.1, 1.15)
    ax.tick_params(axis="x", rotation=25)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")

    p_multi = paired_pvalue(plot_df, "Multi + Pretrained", "Multi + RandomInit")
    p_single = paired_pvalue(plot_df, "Single + Pretrained", "Single + RandomInit")
    annotate_pval(ax, 0, 1, 1.04, 0.05, p_multi)
    annotate_pval(ax, 2, 3, 1.04, 0.05, p_single)

    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    save_fig(fig, FIG_DIR, "A1_violin_cellfm_prior_4branch_per_gene_pearson")
    plt.close(fig)


def plot_a2_ecdf(all_corrs: dict[str, np.ndarray]) -> None:
    thresholds = np.linspace(-0.2, 1.0, 251)
    rows = []

    fig, ax = plt.subplots(figsize=(6.55, 3.45), dpi=300)
    for model_name in ORDER:
        corr = all_corrs[model_name]
        corr = corr[np.isfinite(corr)]
        fractions = np.array([(corr >= threshold).mean() for threshold in thresholds])
        rows.extend(
            {
                "model": model_name,
                "threshold": float(threshold),
                "fraction_genes_ge_threshold": float(frac),
            }
            for threshold, frac in zip(thresholds, fractions)
        )
        ax.plot(thresholds, fractions, color=PALETTE[model_name], linewidth=1.8, label=model_name)

    pd.DataFrame(rows).to_csv(TABLE_DIR / "cellfm_prior_ecdf_threshold_curve.csv", index=False, encoding="utf-8")

    ax.set_title("Fraction of genes above Pearson threshold")
    ax.set_xlabel("Threshold (Pearson r)")
    ax.set_ylabel("Fraction of genes")
    ax.set_xlim(-0.2, 1.0)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=2, loc="upper right", columnspacing=0.9, handlelength=1.8)
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    save_fig(fig, FIG_DIR, "A2_ecdf_fraction_genes_cellfm_prior_4branch")
    plt.close(fig)


def plot_a3_expression_bins(summary_df: pd.DataFrame) -> None:
    bin_plot = summary_df[summary_df["group"].isin(BIN_LABELS)].copy()
    bin_plot["group"] = pd.Categorical(bin_plot["group"], categories=BIN_LABELS, ordered=True)
    bin_plot["model"] = pd.Categorical(bin_plot["model"], categories=ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(6.55, 3.45), dpi=300)
    sns.barplot(
        data=bin_plot,
        x="group",
        y="mean_pearson",
        hue="model",
        hue_order=ORDER,
        palette=[PALETTE[m] for m in ORDER],
        ax=ax,
    )
    ax.set_title("Pearson by true-expression strata")
    ax.set_xlabel(f"True-expression bin from {TEST_ID} true counts")
    ax.set_ylabel("Mean Pearson r")
    ax.legend(frameon=False, ncol=2, loc="upper left", columnspacing=0.8, handlelength=1.4)
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    save_fig(fig, FIG_DIR, "A3_bar_expression_bins_cellfm_prior_4branch")
    plt.close(fig)


def plot_a4_delta(delta_df: pd.DataFrame) -> None:
    delta_plot = delta_df.copy()
    group_order = ["All"] + BIN_LABELS
    delta_plot["group"] = pd.Categorical(delta_plot["group"], categories=group_order, ordered=True)

    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=300)
    sns.pointplot(
        data=delta_plot,
        x="group",
        y="delta_mean_pearson",
        hue="protocol",
        hue_order=["Multi", "Single"],
        palette={"Multi": "#DC0000", "Single": "#3C5488"},
        markers=["o", "s"],
        linestyles=["-", "-"],
        dodge=0.18,
        ax=ax,
    )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title("CellFM prior gain: Pretrained - RandomInit")
    ax.set_xlabel("")
    ax.set_ylabel("Delta mean Pearson r")
    ax.legend(frameon=False, title="Protocol")
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    save_fig(fig, FIG_DIR, "A4_delta_prior_multi_vs_single_cellfm")
    plt.close(fig)


def plot_b1_marker_heatmap(marker_df: pd.DataFrame) -> None:
    marker_keep = [g for g in MARKER_GENES if g in marker_df["gene"].unique()]
    pivot = marker_df.pivot(index="gene", columns="model", values="pearson").loc[marker_keep, ORDER]

    fig, ax = plt.subplots(figsize=(6.55, max(3.2, 0.36 * len(marker_keep))), dpi=300)
    sns.heatmap(
        pivot,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".3f",
        annot_kws={"fontsize": 8.0},
        linewidths=0.5,
        cbar_kws={"label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("Fixed prostate marker panel")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=35)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    plt.tight_layout()
    save_fig(fig, FIG_DIR, "B1_marker_heatmap_cellfm_prior_4branch")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def plot_combined_figure(
    gene_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    marker_df: pd.DataFrame,
    all_corrs: dict[str, np.ndarray],
) -> None:
    """Assemble the four panels corresponding to the manuscript Fig. 4 legend."""
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    width_in = 170 / 25.4
    fig, axes = plt.subplots(2, 2, figsize=(width_in, 4.55), dpi=600)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # Panel a: per-gene Pearson distributions.
    plot_df = gene_df[np.isfinite(gene_df["pearson"])].copy()
    plot_df["model"] = pd.Categorical(plot_df["model"], categories=COMBINED_ORDER, ordered=True)
    sns.violinplot(
        data=plot_df,
        x="model",
        y="pearson",
        order=COMBINED_ORDER,
        hue="model",
        hue_order=COMBINED_ORDER,
        palette=[PALETTE[m] for m in COMBINED_ORDER],
        inner="box",
        cut=0,
        linewidth=0.75,
        legend=False,
        ax=ax_a,
    )
    ax_a.set_title("Per-gene Pearson r")
    ax_a.set_xlabel("")
    ax_a.set_ylabel("Pearson r")
    ax_a.set_ylim(-0.1, 1.15)
    ax_a.set_xticks(range(len(COMBINED_ORDER)))
    ax_a.set_xticklabels(
        [COMBINED_LABELS[m].replace(" ", "\n", 1) for m in COMBINED_ORDER],
        rotation=0,
        fontsize=7.0,
    )
    sns.despine(ax=ax_a, top=True, right=True)

    # Panel b: fraction of genes exceeding Pearson thresholds.
    thresholds = np.linspace(-0.2, 1.0, 251)
    for model_name in COMBINED_ORDER:
        corr = all_corrs[model_name]
        corr = corr[np.isfinite(corr)]
        fractions = np.array([(corr >= threshold).mean() for threshold in thresholds])
        ax_b.plot(thresholds, fractions, color=PALETTE[model_name], linewidth=1.25, label=COMBINED_LABELS[model_name])
    ax_b.set_title("Pearson-threshold curve")
    ax_b.set_xlabel("Pearson r threshold")
    ax_b.set_ylabel("Fraction of genes")
    ax_b.set_xlim(-0.2, 1.0)
    ax_b.set_ylim(0, 1.05)
    handles, labels = ax_b.get_legend_handles_labels()
    legend_idx = [0, 2, 1, 3]
    ax_b.legend(
        [handles[i] for i in legend_idx],
        [labels[i] for i in legend_idx],
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.85,
        ncol=2,
        loc="upper right",
        fontsize=6.5,
        columnspacing=0.7,
        handlelength=1.5,
    )
    sns.despine(ax=ax_b, top=True, right=True)

    # Panel c: Pearson correlation stratified by measured expression.
    bin_plot = summary_df[summary_df["group"].isin(BIN_LABELS)].copy()
    bin_plot["group"] = pd.Categorical(bin_plot["group"], categories=BIN_LABELS, ordered=True)
    bin_plot["model"] = pd.Categorical(bin_plot["model"], categories=COMBINED_ORDER, ordered=True)
    sns.barplot(
        data=bin_plot,
        x="group",
        y="mean_pearson",
        hue="model",
        hue_order=COMBINED_ORDER,
        palette=[PALETTE[m] for m in COMBINED_ORDER],
        ax=ax_c,
    )
    ax_c.set_title("Pearson by expression stratum")
    ax_c.set_xlabel("True-expression stratum")
    ax_c.set_ylabel("Mean Pearson r")
    handles, labels = ax_c.get_legend_handles_labels()
    legend_idx = [0, 2, 1, 3]
    ax_c.legend(
        [handles[i] for i in legend_idx],
        [COMBINED_LABELS[COMBINED_ORDER[i]] for i in legend_idx],
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.85,
        ncol=2,
        loc="upper left",
        fontsize=6.5,
        columnspacing=0.7,
        handlelength=1.5,
    )
    sns.despine(ax=ax_c, top=True, right=True)

    # Panel d: fixed prostate marker panel.
    marker_keep = [g for g in MARKER_GENES if g in marker_df["gene"].unique()]
    pivot = marker_df.pivot(index="gene", columns="model", values="pearson").loc[marker_keep, COMBINED_ORDER]
    sns.heatmap(
        pivot,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".3f",
        annot_kws={"fontsize": 6.5},
        linewidths=0.35,
        cbar_kws={"label": "Pearson r", "shrink": 0.86},
        ax=ax_d,
    )
    ax_d.set_title("Fixed prostate marker genes")
    ax_d.set_xlabel("")
    ax_d.set_ylabel("")
    ax_d.set_xticklabels(["Multi\nPre.", "Single\nPre.", "Multi\nRand.", "Single\nRand."], rotation=0, fontsize=7.0)
    ax_d.set_yticks(np.arange(len(marker_keep)) + 0.5)
    ax_d.set_yticklabels(marker_keep, rotation=0, fontsize=7.5, fontstyle="italic")
    ax_d.tick_params(axis="y", length=0)

    for label, ax in zip("abcd", axes.ravel()):
        panel_label(ax, label)

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.10, top=0.92, wspace=0.34, hspace=0.56)
    for suffix in ["pdf", "png"]:
        out = PAPER_FIG_DIR / f"Figure_4.{suffix}"
        if suffix == "png":
            fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.03)
        else:
            fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved combined Fig. 4 to {PAPER_FIG_DIR}")


def write_readme(summary_df: pd.DataFrame, delta_df: pd.DataFrame) -> None:
    s = summary_df[summary_df["group"] == "All"].set_index("model")
    delta_all = delta_df[delta_df["group"] == "All"].set_index("protocol")
    readme = f"""# CellFM Prior Ablation Nature-Style Figures

Setting: prostate INT28 test. The 2 x 2 ablation compares protocol
(multi-slide vs single-slide) and CellFM initialization (pretrained vs random-init).

| Branch | Mean Pearson | Median Pearson | n genes |
|---|---:|---:|---:|
| Multi + Pretrained | {float(s.loc['Multi + Pretrained', 'mean_pearson']):.6f} | {float(s.loc['Multi + Pretrained', 'median_pearson']):.6f} | {int(s.loc['Multi + Pretrained', 'n_genes'])} |
| Multi + RandomInit | {float(s.loc['Multi + RandomInit', 'mean_pearson']):.6f} | {float(s.loc['Multi + RandomInit', 'median_pearson']):.6f} | {int(s.loc['Multi + RandomInit', 'n_genes'])} |
| Single + Pretrained | {float(s.loc['Single + Pretrained', 'mean_pearson']):.6f} | {float(s.loc['Single + Pretrained', 'median_pearson']):.6f} | {int(s.loc['Single + Pretrained', 'n_genes'])} |
| Single + RandomInit | {float(s.loc['Single + RandomInit', 'mean_pearson']):.6f} | {float(s.loc['Single + RandomInit', 'median_pearson']):.6f} | {int(s.loc['Single + RandomInit', 'n_genes'])} |

Overall prior gain:

- Multi: {float(delta_all.loc['Multi', 'delta_mean_pearson']):+.6f}
- Single: {float(delta_all.loc['Single', 'delta_mean_pearson']):+.6f}

Figure set:

- A1: per-gene Pearson violin plot.
- A2: Pearson-threshold curve.
- A3: true-expression-stratified mean Pearson bar plot.
- A4: pretrained-minus-random-init prior gain.
- B1: fixed prostate marker-gene heatmap.
"""
    (OUTDIR / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    configure_style()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    gene_df, summary_df, delta_df, marker_df, all_corrs = build_tables()
    plot_a1_violin(gene_df)
    plot_a2_ecdf(all_corrs)
    plot_a3_expression_bins(summary_df)
    plot_a4_delta(delta_df)
    plot_b1_marker_heatmap(marker_df)
    plot_combined_figure(gene_df, summary_df, marker_df, all_corrs)
    write_readme(summary_df, delta_df)

    print(f"Saved figures: {FIG_DIR}")
    print(f"Saved tables: {TABLE_DIR}")
    print(summary_df.to_string(index=False))
    print(delta_df.to_string(index=False))


if __name__ == "__main__":
    main()
