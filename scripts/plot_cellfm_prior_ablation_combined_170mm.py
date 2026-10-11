from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "figures_nature_cellfm_prior_combined_170mm"
FIG_DIR = OUTDIR / "figures"
TABLE_DIR = OUTDIR / "tables"

RESULT_DIRS = {
    "Multi + Pretrained": ROOT / "results" / "st_superres" / "cellfm_INT28",
    "Multi + RandomInit": ROOT / "results" / "st_superres" / "cellfm_randominit_ablation" / "multi_seed42_INT28",
    "Single + Pretrained": ROOT / "results" / "prostate_hest" / "morpho" / "single_superres" / "INT25_to_INT28",
    "Single + RandomInit": ROOT / "results" / "st_superres" / "cellfm_randominit_ablation" / "single_seed42_INT28",
}
ORDER = ["Multi + Pretrained", "Multi + RandomInit", "Single + Pretrained", "Single + RandomInit"]
SHORT = {
    "Multi + Pretrained": "Multi pretrained",
    "Multi + RandomInit": "Multi random-init",
    "Single + Pretrained": "Single pretrained",
    "Single + RandomInit": "Single random-init",
}
HEAT_SHORT = {
    "Multi + Pretrained": "Multi\npretrained",
    "Multi + RandomInit": "Multi\nrandom-init",
    "Single + Pretrained": "Single\npretrained",
    "Single + RandomInit": "Single\nrandom-init",
}
PALETTE = {
    "Multi + Pretrained": "#E64B35",
    "Multi + RandomInit": "#F39B7F",
    "Single + Pretrained": "#4DBBD5",
    "Single + RandomInit": "#91D1C2",
}
BIN_LABELS = ["Low", "Mid", "High"]
MARKER_GENES = ["KLK3", "AR", "NKX3-1", "AMACR", "ERG", "EPCAM", "KRT8", "KRT18", "KRT5", "TP63"]


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", rc={"grid.linewidth": 0.38, "grid.alpha": 0.25})
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.2,
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


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_pair(result_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    pred = np.load(require_file(result_dir / "pred_bag.npy"))
    true = np.load(require_file(result_dir / "true_bag.npy"))
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch in {result_dir}: pred={pred.shape}, true={true.shape}")
    return pred.astype(np.float32, copy=False), true.astype(np.float32, copy=False)


def load_gene_list(n_genes: int) -> list[str]:
    candidates = [
        ROOT / "checkpoints" / "st_superres" / "cellfm_forward_sum" / "common_genes.json",
        ROOT / "checkpoints" / "st_superres" / "cellfm_randominit_ablation" / "multi_seed42" / "common_genes.json",
        ROOT / "checkpoints" / "st_superres" / "cellfm_randominit_ablation" / "single_seed42" / "common_genes.json",
        ROOT / "checkpoints" / "st_superres" / "cellfm_single_INT25_to_INT28" / "common_genes.json",
    ]
    for path in candidates:
        if path.exists():
            genes = json.loads(path.read_text(encoding="utf-8"))
            if len(genes) == n_genes:
                return [str(g) for g in genes]
    raise ValueError(f"Cannot find a gene list matching n_genes={n_genes}")


def pearson_per_gene(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred64 = pred.astype(np.float64, copy=False)
    true64 = true.astype(np.float64, copy=False)
    pred_centered = pred64 - pred64.mean(axis=0, keepdims=True)
    true_centered = true64 - true64.mean(axis=0, keepdims=True)
    num = np.sum(pred_centered * true_centered, axis=0)
    den = np.sqrt(np.sum(pred_centered**2, axis=0) * np.sum(true_centered**2, axis=0))
    corr = np.full(pred.shape[1], np.nan, dtype=np.float32)
    valid = den > 0
    corr[valid] = (num[valid] / den[valid]).astype(np.float32)
    return corr


def paired_pvalue(gene_df: pd.DataFrame, left: str, right: str) -> float:
    pivot = gene_df[gene_df["model"].isin([left, right])].pivot(index="gene", columns="model", values="pearson")
    pivot = pivot.replace([np.inf, -np.inf], np.nan).dropna()
    try:
        _, pval = stats.wilcoxon(pivot[left].to_numpy(), pivot[right].to_numpy())
    except Exception:
        _, pval = stats.ttest_rel(pivot[left].to_numpy(), pivot[right].to_numpy())
    return float(pval)


def significance_label(pval: float) -> str:
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return "ns"


def annotate_pval(ax: plt.Axes, idx1: int, idx2: int, y: float, h: float, pval: float) -> None:
    ax.plot([idx1, idx1, idx2, idx2], [y, y + h, y + h, y], lw=0.8, c="0.15", clip_on=False)
    ax.text((idx1 + idx2) * 0.5, y + h + 0.012, significance_label(pval), ha="center", va="bottom", fontsize=8)


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pairs = {}
    ref_true = None
    n_genes = None
    for model, result_dir in RESULT_DIRS.items():
        pred, true = load_pair(result_dir)
        if n_genes is None:
            n_genes = pred.shape[1]
            ref_true = true
        elif pred.shape[1] != n_genes:
            raise ValueError(f"Gene count mismatch for {model}")
        if ref_true is not None and true.shape == ref_true.shape:
            max_abs = float(np.max(np.abs(true - ref_true)))
            if max_abs > 1e-5:
                raise ValueError(f"True matrix differs for {model}; max abs difference={max_abs}")
        pairs[model] = {"pred": pred, "true": true}

    assert n_genes is not None and ref_true is not None
    genes = load_gene_list(n_genes)
    true_gene_intensity = np.log1p(np.clip(ref_true, 0, None)).mean(axis=0)
    q1, q2 = np.quantile(true_gene_intensity, [1 / 3, 2 / 3])
    bin_idx = np.zeros_like(true_gene_intensity, dtype=np.int64)
    bin_idx[true_gene_intensity > q1] = 1
    bin_idx[true_gene_intensity > q2] = 2
    bin_name = np.array(BIN_LABELS, dtype=object)[bin_idx]

    gene_rows = []
    summary_rows = []
    marker_rows = []
    for model in ORDER:
        corr = pearson_per_gene(pairs[model]["pred"], ref_true)
        for idx, gene in enumerate(genes):
            gene_rows.append({"model": model, "gene": gene, "pearson": float(corr[idx]), "expr_bin": str(bin_name[idx])})
        summary_rows.append(
            {
                "model": model,
                "group": "All",
                "mean_pearson": float(np.nanmean(corr)),
                "median_pearson": float(np.nanmedian(corr)),
                "n_genes": int(np.isfinite(corr).sum()),
            }
        )
        for group in BIN_LABELS:
            mask = bin_name == group
            values = corr[mask]
            summary_rows.append(
                {
                    "model": model,
                    "group": group,
                    "mean_pearson": float(np.nanmean(values)),
                    "median_pearson": float(np.nanmedian(values)),
                    "n_genes": int(np.isfinite(values).sum()),
                }
            )
        for gene in MARKER_GENES:
            if gene in genes:
                marker_rows.append({"model": model, "gene": gene, "pearson": float(corr[genes.index(gene)])})

    gene_df = pd.DataFrame(gene_rows)
    summary_df = pd.DataFrame(summary_rows)
    marker_df = pd.DataFrame(marker_rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    gene_df.to_csv(TABLE_DIR / "cellfm_prior_combined_gene_pearson.csv", index=False)
    summary_df.to_csv(TABLE_DIR / "cellfm_prior_combined_summary.csv", index=False)
    marker_df.to_csv(TABLE_DIR / "cellfm_prior_combined_marker_pearson.csv", index=False)
    return gene_df, summary_df, marker_df


def plot_combined(gene_df: pd.DataFrame, summary_df: pd.DataFrame, marker_df: pd.DataFrame) -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    width_in = 170 / 25.4
    fig = plt.figure(figsize=(width_in, 2.35), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.12, 1.18], wspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    plot_df = gene_df[np.isfinite(gene_df["pearson"])].copy()
    plot_df["model"] = pd.Categorical(plot_df["model"], categories=ORDER, ordered=True)
    sns.violinplot(
        data=plot_df,
        x="model",
        y="pearson",
        order=ORDER,
        palette=[PALETTE[m] for m in ORDER],
        inner="box",
        cut=0,
        linewidth=0.65,
        ax=ax_a,
        hue="model",
        legend=False,
    )
    ax_a.set_xlabel("")
    ax_a.set_ylabel("Pearson r")
    ax_a.set_ylim(-0.5, 1.18)
    ax_a.set_xticklabels([])
    ax_a.tick_params(axis="x", length=0)
    p_multi = paired_pvalue(plot_df, "Multi + Pretrained", "Multi + RandomInit")
    p_single = paired_pvalue(plot_df, "Single + Pretrained", "Single + RandomInit")
    annotate_pval(ax_a, 0, 1, 1.07, 0.035, p_multi)
    annotate_pval(ax_a, 2, 3, 1.07, 0.035, p_single)
    handles = [
        mpl.patches.Patch(facecolor=PALETTE[m], edgecolor="0.25", linewidth=0.5, label=SHORT[m])
        for m in ORDER
    ]
    ax_a.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.52, -0.19),
        ncol=2,
        columnspacing=0.85,
        handlelength=1.25,
        handletextpad=0.35,
        borderaxespad=0.0,
    )

    bin_plot = summary_df[summary_df["group"].isin(BIN_LABELS)].copy()
    bin_plot["group"] = pd.Categorical(bin_plot["group"], categories=BIN_LABELS, ordered=True)
    bin_plot["model"] = pd.Categorical(bin_plot["model"], categories=ORDER, ordered=True)
    sns.barplot(
        data=bin_plot,
        x="group",
        y="mean_pearson",
        hue="model",
        hue_order=ORDER,
        palette=[PALETTE[m] for m in ORDER],
        ax=ax_b,
        edgecolor="white",
        linewidth=0.2,
        width=0.78,
    )
    ax_b.set_xlabel("")
    ax_b.set_ylabel("Mean Pearson")
    ax_b.set_ylim(0, 0.78)
    ax_b.legend_.remove()
    ax_b.set_xticklabels(BIN_LABELS)

    marker_keep = [gene for gene in MARKER_GENES if gene in marker_df["gene"].unique()]
    heat = marker_df.pivot(index="gene", columns="model", values="pearson").loc[marker_keep, ORDER]
    sns.heatmap(
        heat,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".3f",
        annot_kws={"fontsize": 5.8},
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "Pearson r", "shrink": 0.72, "pad": 0.02, "ticks": [-1, -0.5, 0, 0.5, 1]},
        ax=ax_c,
    )
    ax_c.set_xlabel("")
    ax_c.set_ylabel("")
    ax_c.set_xticklabels([HEAT_SHORT[m] for m in ORDER], rotation=35, ha="right", fontsize=6.8)
    ax_c.set_yticklabels(ax_c.get_yticklabels(), rotation=0, fontsize=7.2)
    ax_c.collections[0].colorbar.ax.tick_params(labelsize=6.8, length=2)
    ax_c.collections[0].colorbar.set_label("Pearson r", fontsize=7.2)

    for ax in [ax_a, ax_b]:
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", alpha=0.22, linewidth=0.38)
        sns.despine(ax=ax, top=True, right=True)

    for label, ax in zip(["a", "b", "c"], [ax_a, ax_b, ax_c]):
        ax.text(-0.22, 1.09, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.27, top=0.88, wspace=0.34)
    fig.savefig(FIG_DIR / "cellfm_prior_ablation_combined_170mm.pdf")
    fig.savefig(FIG_DIR / "cellfm_prior_ablation_combined_170mm.svg")
    fig.savefig(FIG_DIR / "cellfm_prior_ablation_combined_170mm.png", dpi=600)
    plt.close(fig)


def main() -> None:
    gene_df, summary_df, marker_df = build_tables()
    plot_combined(gene_df, summary_df, marker_df)
    print(summary_df[summary_df["group"].eq("All")].to_string(index=False))
    print(f"Saved figure to {FIG_DIR}")
    print(f"Saved tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
