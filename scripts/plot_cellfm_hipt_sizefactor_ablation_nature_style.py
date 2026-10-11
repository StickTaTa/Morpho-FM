from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(os.environ.get("MORPHO_VC_ROOT", Path.cwd())).expanduser().resolve()
if not (ROOT / "results").exists() and (ROOT.parent / "results").exists():
    ROOT = ROOT.parent

RESULT_ROOT = ROOT / "results" / "st_superres" / "cellfm_hipt_sizefactor_ablation_v1"
OUTDIR = ROOT / "figures_nature_cellfm_hipt_sizefactor_ablation"
FIG_DIR = OUTDIR / "figures"
TABLE_DIR = OUTDIR / "tables"

BRANCHES = {
    "Single | Pretrained": {
        "protocol": "single",
        "decoder": "Pretrained CellFM",
        "path": RESULT_ROOT / "single_pretrained_INT28" / "INT28",
        "fold": RESULT_ROOT / "single_pretrained_INT28" / "fold_summary_hipt_sizefactor.csv",
        "color": "#4DBBD5",
    },
    "Single | Random": {
        "protocol": "single",
        "decoder": "Random CellFM",
        "path": RESULT_ROOT / "single_random_INT28" / "INT28",
        "fold": RESULT_ROOT / "single_random_INT28" / "fold_summary_hipt_sizefactor.csv",
        "color": "#91D1C2",
    },
    "Multi | Pretrained": {
        "protocol": "multi",
        "decoder": "Pretrained CellFM",
        "path": RESULT_ROOT / "multi_pretrained_INT28" / "INT28",
        "fold": RESULT_ROOT / "multi_pretrained_INT28" / "fold_summary_hipt_sizefactor.csv",
        "color": "#E64B35",
    },
    "Multi | Random": {
        "protocol": "multi",
        "decoder": "Random CellFM",
        "path": RESULT_ROOT / "multi_random_INT28" / "INT28",
        "fold": RESULT_ROOT / "multi_random_INT28" / "fold_summary_hipt_sizefactor.csv",
        "color": "#F39B7F",
    },
}
ORDER = list(BRANCHES)
MARKER_GENES = ["KLK3", "KLK2", "AR", "NKX3-1", "AMACR", "ERG", "EPCAM", "KRT8", "KRT18", "TP63"]
BIN_LABELS = ["Low", "Mid", "High"]


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
            "legend.fontsize": 7,
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
        ROOT / "checkpoints" / "st_superres" / "cellfm_single_INT25_to_INT28" / "common_genes.json",
    ]
    for path in candidates:
        if path.exists():
            genes = json.loads(path.read_text(encoding="utf-8"))
            if len(genes) == n_genes:
                return [str(g) for g in genes]
    raise ValueError(f"No gene list found for {n_genes} genes.")


def pearson_per_gene(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    true64 = true.astype(np.float64, copy=False)
    pred64 = pred.astype(np.float64, copy=False)
    true_centered = true64 - true64.mean(axis=0, keepdims=True)
    pred_centered = pred64 - pred64.mean(axis=0, keepdims=True)
    num = np.sum(true_centered * pred_centered, axis=0)
    den = np.sqrt(np.sum(true_centered**2, axis=0) * np.sum(pred_centered**2, axis=0))
    out = np.full(true.shape[1], np.nan, dtype=np.float32)
    valid = den > 0
    out[valid] = (num[valid] / den[valid]).astype(np.float32)
    return out


def expression_bins(true: np.ndarray) -> np.ndarray:
    mean_expr = np.log1p(np.clip(true, 0, None)).mean(axis=0)
    q1, q2 = np.quantile(mean_expr, [1 / 3, 2 / 3])
    bins = np.zeros(mean_expr.shape[0], dtype=int)
    bins[mean_expr > q1] = 1
    bins[mean_expr > q2] = 2
    return bins


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pairs = {}
    n_genes = None
    ref_true = None
    for name, spec in BRANCHES.items():
        pred, true = load_pair(spec["path"])
        if n_genes is None:
            n_genes = pred.shape[1]
            ref_true = true
        elif pred.shape[1] != n_genes:
            raise ValueError(f"Gene count mismatch for {name}: {pred.shape[1]} vs {n_genes}")
        pairs[name] = (pred, true)

    assert n_genes is not None and ref_true is not None
    genes = load_genes(n_genes)
    bins = expression_bins(ref_true)
    summary_rows = []
    gene_rows = []
    strata_rows = []
    marker_rows = []
    sf_rows = []
    for name in ORDER:
        spec = BRANCHES[name]
        pred, true = pairs[name]
        corr = pearson_per_gene(true, pred)
        true_var = np.var(true, axis=0)
        hvg_idx = np.argsort(true_var)[::-1][: min(1000, n_genes)]
        fold = pd.read_csv(require_file(spec["fold"])).iloc[0].to_dict()
        sf_metrics = json.loads((spec["path"] / "sf_estimator_metrics.json").read_text(encoding="utf-8"))
        summary_rows.append(
            {
                "branch": name,
                "protocol": spec["protocol"],
                "decoder": spec["decoder"],
                "mean_pearson": float(np.nanmean(corr)),
                "median_pearson": float(np.nanmedian(corr)),
                "hvg1000_mean_pearson": float(np.nanmean(corr[hvg_idx])),
                "rate_mean_pearson": float(fold["Rate_Mean_Pearson"]),
                "sf_pearson": float(sf_metrics["sf_pearson"]),
                "log_sf_pearson": float(sf_metrics["log_sf_pearson"]),
                "n_genes": int(np.isfinite(corr).sum()),
                "n_spots": int(pred.shape[0]),
            }
        )
        sf_rows.append({"branch": name, **sf_metrics})
        for gene, value, bin_idx in zip(genes, corr, bins):
            gene_rows.append(
                {
                    "branch": name,
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
                    "branch": name,
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
                        "branch": name,
                        "protocol": spec["protocol"],
                        "decoder": spec["decoder"],
                        "gene": gene,
                        "pearson": float(corr[idx]),
                    }
                )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(gene_rows),
        pd.DataFrame(strata_rows),
        pd.DataFrame(marker_rows),
        pd.DataFrame(sf_rows),
    )


def save_outputs(summary: pd.DataFrame, genes: pd.DataFrame, strata: pd.DataFrame, markers: pd.DataFrame, sf: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE_DIR / "cellfm_hipt_sizefactor_summary.csv", index=False)
    genes.to_csv(TABLE_DIR / "cellfm_hipt_sizefactor_gene_pearson.csv", index=False)
    strata.to_csv(TABLE_DIR / "cellfm_hipt_sizefactor_expression_strata.csv", index=False)
    markers.to_csv(TABLE_DIR / "cellfm_hipt_sizefactor_marker_pearson.csv", index=False)
    sf.to_csv(TABLE_DIR / "cellfm_hipt_sizefactor_sf_metrics.csv", index=False)


def plot_figure(summary: pd.DataFrame, genes: pd.DataFrame, strata: pd.DataFrame, markers: pd.DataFrame) -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    palette = {name: BRANCHES[name]["color"] for name in ORDER}

    fig = plt.figure(figsize=(6.65, 7.45), constrained_layout=False)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.22, 1.25], hspace=0.70, wspace=0.42)

    ax_a = fig.add_subplot(gs[0, 0])
    sorted_summary = summary.sort_values("mean_pearson", ascending=True).reset_index(drop=True)
    y_labels = [x.replace(" | ", " ") for x in sorted_summary["branch"]]
    bars = ax_a.barh(
        y_labels,
        sorted_summary["mean_pearson"],
        color=[palette[b] for b in sorted_summary["branch"]],
        edgecolor="0.25",
        linewidth=0.7,
    )
    max_mean = float(summary["mean_pearson"].max())
    ax_a.set_xlim(0, max(0.22, max_mean + 0.035))
    ax_a.set_xlabel("Mean r")
    ax_a.set_ylabel("")
    ax_a.set_title("HIPT-size-factor prediction, ranked")
    ax_a.bar_label(bars, fmt="%.3f", fontsize=6.8, padding=2)

    ax_b = fig.add_subplot(gs[0, 1])
    delta_rows = []
    for protocol in ["single", "multi"]:
        sub = summary[summary["protocol"].eq(protocol)].set_index("decoder")
        delta_rows.append(
            {
                "protocol": protocol,
                "comparison": "PT - Random",
                "delta": float(sub.loc["Pretrained CellFM", "mean_pearson"] - sub.loc["Random CellFM", "mean_pearson"]),
            }
        )
    delta_df = pd.DataFrame(delta_rows)
    sns.barplot(data=delta_df, x="protocol", y="delta", color="#9ABFA8", ax=ax_b, edgecolor="0.25", linewidth=0.7)
    ax_b.axhline(0, color="0.2", linewidth=0.8)
    ax_b.set_xlabel("")
    ax_b.set_ylabel("Delta mean r")
    ax_b.set_title("Pretrained gain over random")
    ax_b.set_ylim(0, max(0.13, float(delta_df["delta"].max()) + 0.025))
    for container in ax_b.containers:
        ax_b.bar_label(container, fmt="%+.3f", fontsize=6.8, padding=2)

    ax_c = fig.add_subplot(gs[1, :])
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
        width=0.18,
        boxprops={"facecolor": "white", "edgecolor": "0.25", "linewidth": 0.7},
        whiskerprops={"linewidth": 0.7, "color": "0.25"},
        capprops={"linewidth": 0.7, "color": "0.25"},
        medianprops={"linewidth": 0.8, "color": "0.1"},
        showfliers=False,
    )
    ax_c.set_xlabel("")
    ax_c.set_ylabel("Per-gene r")
    ax_c.set_title("Genome-wide distributions on INT28")
    ax_c.set_ylim(-0.35, 1.05)
    ax_c.set_xticklabels(["Single\nPT", "Single\nRI", "Multi\nPT", "Multi\nRI"], rotation=0)

    ax_d = fig.add_subplot(gs[2, 0])
    sns.lineplot(
        data=strata,
        x="expression_bin",
        y="mean_pearson",
        hue="branch",
        hue_order=ORDER,
        palette=palette,
        marker="o",
        linewidth=1.5,
        markersize=5,
        ax=ax_d,
    )
    ax_d.set_xlabel("Expression stratum")
    ax_d.set_ylabel("Mean r")
    ax_d.set_title("Expression-stratified performance")
    max_strata = float(strata["mean_pearson"].max())
    ax_d.set_ylim(0, max(0.56, max_strata + 0.05))
    ax_d.legend(frameon=False, title=None, fontsize=6.2, loc="upper left", bbox_to_anchor=(0, 1.02))

    ax_e = fig.add_subplot(gs[2, 1])
    heat = markers.pivot(index="gene", columns="branch", values="pearson").reindex(MARKER_GENES)[ORDER]
    sns.heatmap(
        heat,
        cmap="viridis",
        vmin=0,
        vmax=1,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Pearson r", "shrink": 0.78},
        ax=ax_e,
    )
    ax_e.set_xlabel("")
    ax_e.set_ylabel("")
    ax_e.set_title("Fixed prostate markers")
    ax_e.set_xticklabels(["S PT", "S RI", "M PT", "M RI"], rotation=45, ha="right", fontsize=7)
    ax_e.set_yticklabels(ax_e.get_yticklabels(), fontsize=7)

    for label, ax in zip(["a", "b", "c", "d", "e"], [ax_a, ax_b, ax_c, ax_d, ax_e]):
        ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")

    fig.savefig(FIG_DIR / "cellfm_hipt_sizefactor_ablation_supplement.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "cellfm_hipt_sizefactor_ablation_supplement.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / "cellfm_hipt_sizefactor_ablation_supplement.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def write_text(summary: pd.DataFrame) -> None:
    rows = summary.set_index(["protocol", "decoder"])
    text = f"""HIPT-estimated size-factor CellFM ablation summary

Single-slide INT25-to-INT28:
- Pretrained CellFM count-scale mean Pearson: {rows.loc[('single', 'Pretrained CellFM'), 'mean_pearson']:.4f}
- Random-init CellFM count-scale mean Pearson: {rows.loc[('single', 'Random CellFM'), 'mean_pearson']:.4f}
- Pretrained minus random: {rows.loc[('single', 'Pretrained CellFM'), 'mean_pearson'] - rows.loc[('single', 'Random CellFM'), 'mean_pearson']:+.4f}
- INT28 image-estimated size-factor log-scale Pearson: {rows.loc[('single', 'Pretrained CellFM'), 'log_sf_pearson']:.4f}

Multi-slide INT25+INT26 train, INT27 validation, INT28 test:
- Pretrained CellFM count-scale mean Pearson: {rows.loc[('multi', 'Pretrained CellFM'), 'mean_pearson']:.4f}
- Random-init CellFM count-scale mean Pearson: {rows.loc[('multi', 'Random CellFM'), 'mean_pearson']:.4f}
- Pretrained minus random: {rows.loc[('multi', 'Pretrained CellFM'), 'mean_pearson'] - rows.loc[('multi', 'Random CellFM'), 'mean_pearson']:+.4f}
- INT28 image-estimated size-factor log-scale Pearson: {rows.loc[('multi', 'Pretrained CellFM'), 'log_sf_pearson']:.4f}
"""
    (TABLE_DIR / "cellfm_hipt_sizefactor_interpretation.txt").write_text(text, encoding="utf-8")


def main() -> None:
    summary, genes, strata, markers, sf = load_all()
    save_outputs(summary, genes, strata, markers, sf)
    plot_figure(summary, genes, strata, markers)
    write_text(summary)
    print(summary)
    print(f"Saved figure to {FIG_DIR}")
    print(f"Saved tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
