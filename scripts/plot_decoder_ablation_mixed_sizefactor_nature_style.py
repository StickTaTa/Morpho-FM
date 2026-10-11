from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "figures_nature_decoder_ablation_mixed_sizefactor"
FIG_DIR = OUTDIR / "figures"
TABLE_DIR = OUTDIR / "tables"

BRANCHES = {
    "Single | PT CellFM (oracle SF)": {
        "protocol": "single",
        "decoder": "PT CellFM (oracle SF)",
        "short": "Single\nPT oracle",
        "path": ROOT / "results" / "prostate_hest" / "morpho" / "single_superres" / "INT25_to_INT28",
        "color": "#4DBBD5",
    },
    "Single | Random CellFM (rate)": {
        "protocol": "single",
        "decoder": "Random CellFM (rate)",
        "short": "Single\nRI rate",
        "path": ROOT / "results" / "st_superres" / "cellfm_clean_rate_ablation" / "single_random_INT28" / "INT28",
        "color": "#91D1C2",
    },
    "Single | MLP (rate)": {
        "protocol": "single",
        "decoder": "MLP (rate)",
        "short": "Single\nMLP rate",
        "path": ROOT / "results" / "ablation_decoder" / "mlp_decoder_clean_rate_single_INT25_to_INT28" / "INT28",
        "color": "#7E8CE0",
    },
    "Multi | PT CellFM (oracle SF)": {
        "protocol": "multi",
        "decoder": "PT CellFM (oracle SF)",
        "short": "Multi\nPT oracle",
        "path": ROOT / "results" / "st_superres" / "cellfm_INT28",
        "color": "#E64B35",
    },
    "Multi | Random CellFM (rate)": {
        "protocol": "multi",
        "decoder": "Random CellFM (rate)",
        "short": "Multi\nRI rate",
        "path": ROOT / "results" / "st_superres" / "cellfm_clean_rate_ablation" / "multi_random_INT28" / "INT28",
        "color": "#F39B7F",
    },
    "Multi | MLP (rate)": {
        "protocol": "multi",
        "decoder": "MLP (rate)",
        "short": "Multi\nMLP rate",
        "path": ROOT / "results" / "ablation_decoder" / "mlp_decoder_clean_rate_multi_INT25_INT26_val_INT27_to_INT28" / "INT28",
        "color": "#B07AA1",
    },
}
ORDER = list(BRANCHES)
BIN_LABELS = ["Low", "Mid", "High"]
MARKER_GENES = ["KLK3", "KLK2", "AR", "NKX3-1", "AMACR", "ERG", "EPCAM", "KRT8", "KRT18", "TP63"]


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", rc={"grid.linewidth": 0.45, "grid.alpha": 0.22})
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 6.8,
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


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_pair(result_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    pred = np.load(require_file(result_dir / "pred_bag.npy"))
    true = np.load(require_file(result_dir / "true_bag.npy"))
    if pred.shape != true.shape:
        n = min(pred.shape[0], true.shape[0])
        pred = pred[:n]
        true = true[:n]
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch in {result_dir}: pred={pred.shape}, true={true.shape}")
    return pred.astype(np.float32, copy=False), true.astype(np.float32, copy=False)


def load_genes(n_genes: int) -> list[str]:
    candidates = [
        ROOT / "checkpoints" / "st_superres" / "cellfm_forward_sum" / "common_genes.json",
        ROOT / "checkpoints" / "st_superres" / "cellfm_randominit_ablation" / "multi_seed42" / "common_genes.json",
        ROOT / "checkpoints" / "st_superres" / "cellfm_randominit_ablation" / "single_seed42" / "common_genes.json",
        ROOT / "checkpoints" / "st_superres" / "decoder_ablation" / "mlp_decoder_single_INT25_to_INT28" / "common_genes.json",
    ]
    for path in candidates:
        if path.exists():
            genes = json.loads(path.read_text(encoding="utf-8"))
            if len(genes) == n_genes:
                return [str(g) for g in genes]
    raise ValueError(f"No gene list found for {n_genes} genes")


def pearson_per_gene(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    true64 = true.astype(np.float64, copy=False)
    pred64 = pred.astype(np.float64, copy=False)
    true_centered = true64 - true64.mean(axis=0, keepdims=True)
    pred_centered = pred64 - pred64.mean(axis=0, keepdims=True)
    num = np.sum(true_centered * pred_centered, axis=0)
    den = np.sqrt(np.sum(true_centered**2, axis=0) * np.sum(pred_centered**2, axis=0))
    corr = np.full(true.shape[1], np.nan, dtype=np.float32)
    valid = den > 0
    corr[valid] = (num[valid] / den[valid]).astype(np.float32)
    return corr


def expression_bins(true: np.ndarray) -> np.ndarray:
    mean_expr = np.log1p(np.clip(true, 0, None)).mean(axis=0)
    q1, q2 = np.quantile(mean_expr, [1 / 3, 2 / 3])
    bins = np.zeros(mean_expr.shape[0], dtype=int)
    bins[mean_expr > q1] = 1
    bins[mean_expr > q2] = 2
    return bins


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    gene_rows = []
    strata_rows = []
    marker_rows = []
    n_genes = None
    genes: list[str] | None = None
    for branch in ORDER:
        spec = BRANCHES[branch]
        pred, true = load_pair(spec["path"])
        if n_genes is None:
            n_genes = pred.shape[1]
            genes = load_genes(n_genes)
        elif pred.shape[1] != n_genes:
            raise ValueError(f"Gene count mismatch for {branch}: {pred.shape[1]} vs {n_genes}")
        assert genes is not None
        corr = pearson_per_gene(true, pred)
        true_var = np.var(true, axis=0)
        hvg_idx = np.argsort(true_var)[::-1][: min(1000, pred.shape[1])]
        bins = expression_bins(true)
        valid = np.isfinite(corr)
        summary_rows.append(
            {
                "branch": branch,
                "protocol": spec["protocol"],
                "decoder": spec["decoder"],
                "mean_pearson": float(np.nanmean(corr)),
                "median_pearson": float(np.nanmedian(corr)),
                "hvg1000_mean_pearson": float(np.nanmean(corr[hvg_idx])),
                "n_genes": int(valid.sum()),
                "n_spots": int(pred.shape[0]),
                "target_scale": "count/oracle_sf" if "oracle SF" in spec["decoder"] else "normalized_rate",
            }
        )
        for gene, value, bin_idx in zip(genes, corr, bins):
            gene_rows.append(
                {
                    "branch": branch,
                    "protocol": spec["protocol"],
                    "decoder": spec["decoder"],
                    "gene": gene,
                    "pearson": float(value),
                    "expression_bin": BIN_LABELS[int(bin_idx)],
                }
            )
        for bin_idx, bin_label in enumerate(BIN_LABELS):
            mask = bins == bin_idx
            strata_rows.append(
                {
                    "branch": branch,
                    "protocol": spec["protocol"],
                    "decoder": spec["decoder"],
                    "expression_bin": bin_label,
                    "mean_pearson": float(np.nanmean(corr[mask])),
                    "n_genes": int(mask.sum()),
                }
            )
        for gene in MARKER_GENES:
            if gene in genes:
                idx = genes.index(gene)
                marker_rows.append(
                    {
                        "branch": branch,
                        "protocol": spec["protocol"],
                        "decoder": spec["decoder"],
                        "gene": gene,
                        "pearson": float(corr[idx]),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(gene_rows), pd.DataFrame(strata_rows), pd.DataFrame(marker_rows)


def save_tables(summary: pd.DataFrame, genes: pd.DataFrame, strata: pd.DataFrame, markers: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE_DIR / "decoder_ablation_mixed_sizefactor_summary.csv", index=False)
    genes.to_csv(TABLE_DIR / "decoder_ablation_mixed_sizefactor_gene_pearson.csv", index=False)
    strata.to_csv(TABLE_DIR / "decoder_ablation_mixed_sizefactor_expression_strata.csv", index=False)
    markers.to_csv(TABLE_DIR / "decoder_ablation_mixed_sizefactor_marker_pearson.csv", index=False)


def plot_figure(summary: pd.DataFrame, genes: pd.DataFrame, strata: pd.DataFrame, markers: pd.DataFrame) -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    palette = {branch: BRANCHES[branch]["color"] for branch in ORDER}

    fig = plt.figure(figsize=(6.65, 6.75), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.35], hspace=0.62, wspace=0.44)

    ax_a = fig.add_subplot(gs[0, 0])
    ranked = summary.sort_values("mean_pearson", ascending=True).reset_index(drop=True)
    labels = [
        BRANCHES[branch]["short"].replace("\n", " ")
        for branch in ranked["branch"]
    ]
    bars = ax_a.barh(
        labels,
        ranked["mean_pearson"],
        color=[palette[branch] for branch in ranked["branch"]],
        edgecolor="0.25",
        linewidth=0.7,
    )
    ax_a.set_xlim(0, max(0.40, float(summary["mean_pearson"].max()) + 0.045))
    ax_a.set_xlabel("Mean r")
    ax_a.set_ylabel("")
    ax_a.set_title("Mixed size-factor decoder comparison")
    ax_a.bar_label(bars, fmt="%.3f", fontsize=6.7, padding=2)

    ax_b = fig.add_subplot(gs[0, 1])
    sns.lineplot(
        data=strata,
        x="expression_bin",
        y="mean_pearson",
        hue="branch",
        hue_order=ORDER,
        palette=palette,
        marker="o",
        linewidth=1.5,
        markersize=4.8,
        ax=ax_b,
    )
    ax_b.set_xlabel("True-expression stratum")
    ax_b.set_ylabel("Mean r")
    ax_b.set_title("Pearson by true-expression strata")
    ax_b.set_ylim(0, max(0.95, float(strata["mean_pearson"].max()) + 0.06))
    handles, labels = ax_b.get_legend_handles_labels()
    short_labels = [BRANCHES[label]["short"].replace("\n", " ") if label in BRANCHES else label for label in labels]
    ax_b.legend(handles, short_labels, frameon=False, title=None, loc="upper left", bbox_to_anchor=(0.0, 1.02), fontsize=5.8)

    ax_c = fig.add_subplot(gs[1, 0])
    sns.violinplot(
        data=genes,
        x="branch",
        y="pearson",
        order=ORDER,
        palette=palette,
        ax=ax_c,
        cut=0,
        inner=None,
        linewidth=0.6,
        saturation=0.95,
    )
    sns.boxplot(
        data=genes,
        x="branch",
        y="pearson",
        order=ORDER,
        ax=ax_c,
        showcaps=True,
        width=0.20,
        boxprops={"facecolor": "white", "edgecolor": "0.25", "linewidth": 0.7},
        whiskerprops={"linewidth": 0.7, "color": "0.25"},
        capprops={"linewidth": 0.7, "color": "0.25"},
        medianprops={"linewidth": 0.8, "color": "0.1"},
        showfliers=False,
    )
    ax_c.set_xlabel("")
    ax_c.set_ylabel("Per-gene r")
    ax_c.set_title("Genome-wide distributions")
    ax_c.set_ylim(-0.35, 1.05)
    ax_c.set_xticklabels(["S PT", "S RI", "S MLP", "M PT", "M RI", "M MLP"], rotation=0, fontsize=7)

    ax_d = fig.add_subplot(gs[1, 1])
    heat = markers.pivot(index="gene", columns="branch", values="pearson").reindex(MARKER_GENES)[ORDER]
    sns.heatmap(
        heat,
        cmap="viridis",
        vmin=0,
        vmax=1,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Pearson r", "shrink": 0.74},
        ax=ax_d,
    )
    ax_d.set_xlabel("")
    ax_d.set_ylabel("")
    ax_d.set_title("Fixed prostate markers")
    ax_d.set_xticklabels(["S PT", "S RI", "S MLP", "M PT", "M RI", "M MLP"], rotation=45, ha="right", fontsize=6.8)
    ax_d.set_yticklabels(ax_d.get_yticklabels(), fontsize=7)

    for label, ax in zip(["a", "b", "c", "d"], [ax_a, ax_b, ax_c, ax_d]):
        ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")

    fig.savefig(FIG_DIR / "decoder_ablation_mixed_sizefactor_supplement.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "decoder_ablation_mixed_sizefactor_supplement.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / "decoder_ablation_mixed_sizefactor_supplement.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    summary, genes, strata, markers = load_all()
    save_tables(summary, genes, strata, markers)
    plot_figure(summary, genes, strata, markers)
    print(summary)
    print(f"Saved figure to {FIG_DIR}")
    print(f"Saved tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
