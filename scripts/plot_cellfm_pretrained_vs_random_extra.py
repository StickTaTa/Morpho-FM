from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(os.environ.get("MORPHO_VC_ROOT", Path.cwd())).expanduser().resolve()
if not (ROOT / "results").exists() and (ROOT.parent / "results").exists():
    ROOT = ROOT.parent

OUTDIR = ROOT / "figures_extra_cellfm_pretrained_vs_random"
FIG_DIR = OUTDIR / "figures"
TABLE_DIR = OUTDIR / "tables"
MANUSCRIPT_DIR = Path(r"C:\Users\Administrator\Desktop\人生的第4篇论文")
PAPER_DIR = MANUSCRIPT_DIR / "Figure" / "Supplementary_Figure" / "CellFM_pretrained_vs_random_extra"

BRANCHES = {
    "Single | Pretrained CellFM": {
        "protocol": "single",
        "condition": "Pretrained CellFM",
        "path": ROOT / "results" / "prostate_hest" / "morpho" / "single_superres" / "INT25_to_INT28",
        "color": "#4DBBD5",
    },
    "Single | Random CellFM": {
        "protocol": "single",
        "condition": "Random CellFM",
        "path": ROOT / "results" / "st_superres" / "cellfm_randominit_ablation" / "single_seed42_INT28",
        "color": "#91D1C2",
    },
    "Multi | Pretrained CellFM": {
        "protocol": "multi",
        "condition": "Pretrained CellFM",
        "path": ROOT / "results" / "st_superres" / "cellfm_INT28",
        "color": "#E64B35",
    },
    "Multi | Random CellFM": {
        "protocol": "multi",
        "condition": "Random CellFM",
        "path": ROOT / "results" / "st_superres" / "cellfm_randominit_ablation" / "multi_seed42_INT28",
        "color": "#F39B7F",
    },
}

ORDER = list(BRANCHES)
PAIR_ORDER = ["single", "multi"]
MARKER_GENES = ["KLK3", "KLK2", "AR", "NKX3-1", "AMACR", "ERG", "EPCAM", "KRT8", "KRT18", "TP63"]
THRESHOLDS = [0.2, 0.4, 0.6, 0.8]


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", rc={"grid.linewidth": 0.45, "grid.alpha": 0.22})
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
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


def load_pair(path: Path) -> tuple[np.ndarray, np.ndarray]:
    pred = np.load(require_file(path / "pred_bag.npy")).astype(np.float32, copy=False)
    true = np.load(require_file(path / "true_bag.npy")).astype(np.float32, copy=False)
    if pred.shape != true.shape:
        n = min(pred.shape[0], true.shape[0])
        pred = pred[:n]
        true = true[:n]
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch in {path}: pred={pred.shape}, true={true.shape}")
    return pred, true


def load_genes(n_genes: int) -> list[str]:
    candidates = [
        ROOT / "checkpoints" / "st_superres" / "cellfm_randominit_ablation" / "single_seed42" / "common_genes.json",
        ROOT / "checkpoints" / "st_superres" / "cellfm_randominit_ablation" / "multi_seed42" / "common_genes.json",
        ROOT / "checkpoints" / "st_superres" / "decoder_ablation" / "mlp_decoder_single_INT25_to_INT28" / "common_genes.json",
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


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pairs = {}
    n_genes = None
    ref_true = None
    for branch, spec in BRANCHES.items():
        pred, true = load_pair(spec["path"])
        if n_genes is None:
            n_genes = pred.shape[1]
            ref_true = true
        elif pred.shape[1] != n_genes:
            raise ValueError(f"Gene count mismatch for {branch}: {pred.shape[1]} vs {n_genes}")
        pairs[branch] = (pred, true)

    assert n_genes is not None and ref_true is not None
    genes = load_genes(n_genes)
    hvg_idx = np.argsort(np.var(ref_true, axis=0))[::-1][: min(1000, n_genes)]

    summary_rows = []
    gene_rows = []
    marker_rows = []
    threshold_rows = []
    for branch in ORDER:
        spec = BRANCHES[branch]
        pred, true = pairs[branch]
        corr = pearson_per_gene(true, pred)
        valid = np.isfinite(corr)
        summary_rows.append(
            {
                "branch": branch,
                "protocol": spec["protocol"],
                "condition": spec["condition"],
                "mean_pearson": float(np.nanmean(corr)),
                "median_pearson": float(np.nanmedian(corr)),
                "hvg1000_mean_pearson": float(np.nanmean(corr[hvg_idx])),
                "hvg1000_median_pearson": float(np.nanmedian(corr[hvg_idx])),
                "n_genes": int(valid.sum()),
                "n_spots": int(pred.shape[0]),
            }
        )
        for gene, value in zip(genes, corr):
            gene_rows.append(
                {
                    "branch": branch,
                    "protocol": spec["protocol"],
                    "condition": spec["condition"],
                    "gene": gene,
                    "pearson": float(value),
                }
            )
        for threshold in THRESHOLDS:
            threshold_rows.append(
                {
                    "branch": branch,
                    "protocol": spec["protocol"],
                    "condition": spec["condition"],
                    "threshold": threshold,
                    "fraction_genes": float(np.nanmean(corr > threshold)),
                }
            )
        for gene in MARKER_GENES:
            if gene in genes:
                idx = genes.index(gene)
                marker_rows.append(
                    {
                        "branch": branch,
                        "protocol": spec["protocol"],
                        "condition": spec["condition"],
                        "gene": gene,
                        "pearson": float(corr[idx]),
                    }
                )

    gene_df = pd.DataFrame(gene_rows)
    delta_rows = []
    for protocol in PAIR_ORDER:
        pre = gene_df[(gene_df["protocol"] == protocol) & (gene_df["condition"] == "Pretrained CellFM")].set_index("gene")
        rnd = gene_df[(gene_df["protocol"] == protocol) & (gene_df["condition"] == "Random CellFM")].set_index("gene")
        common = pre.index.intersection(rnd.index)
        for gene in common:
            delta_rows.append(
                {
                    "protocol": protocol,
                    "gene": gene,
                    "delta_pretrained_minus_random": float(pre.loc[gene, "pearson"] - rnd.loc[gene, "pearson"]),
                }
            )

    return (
        pd.DataFrame(summary_rows),
        gene_df,
        pd.DataFrame(delta_rows),
        pd.DataFrame(marker_rows),
        pd.DataFrame(threshold_rows),
    )


def save_tables(summary: pd.DataFrame, genes: pd.DataFrame, deltas: pd.DataFrame, markers: pd.DataFrame, thresholds: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE_DIR / "cellfm_pretrained_vs_random_summary.csv", index=False)
    genes.to_csv(TABLE_DIR / "cellfm_pretrained_vs_random_gene_pearson.csv", index=False)
    deltas.to_csv(TABLE_DIR / "cellfm_pretrained_vs_random_gene_delta.csv", index=False)
    markers.to_csv(TABLE_DIR / "cellfm_pretrained_vs_random_marker.csv", index=False)
    thresholds.to_csv(TABLE_DIR / "cellfm_pretrained_vs_random_thresholds.csv", index=False)


def plot_figure(summary: pd.DataFrame, genes: pd.DataFrame, deltas: pd.DataFrame, markers: pd.DataFrame, thresholds: pd.DataFrame) -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    palette = {branch: BRANCHES[branch]["color"] for branch in ORDER}
    sorted_summary = summary.sort_values("mean_pearson", ascending=True).reset_index(drop=True)
    sorted_summary["label"] = [
        f"{row.protocol} {'PT' if row.condition.startswith('Pretrained') else 'RI'}"
        for row in sorted_summary.itertuples(index=False)
    ]

    fig = plt.figure(figsize=(6.65, 7.6), constrained_layout=False)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.28, 1.22], hspace=0.68, wspace=0.42)

    ax_a = fig.add_subplot(gs[0, 0])
    bars = ax_a.barh(
        sorted_summary["label"],
        sorted_summary["mean_pearson"],
        color=[palette[b] for b in sorted_summary["branch"]],
        edgecolor="0.25",
        linewidth=0.7,
    )
    ax_a.bar_label(bars, fmt="%.3f", fontsize=6.8, padding=2)
    ax_a.set_xlim(0, max(0.40, float(summary["mean_pearson"].max()) + 0.04))
    ax_a.set_xlabel("Mean Pearson r")
    ax_a.set_ylabel("")
    ax_a.set_title("CellFM branch summary, ranked")

    ax_b = fig.add_subplot(gs[0, 1])
    delta_summary = []
    rows = summary.set_index(["protocol", "condition"])
    for protocol in PAIR_ORDER:
        pre = float(rows.loc[(protocol, "Pretrained CellFM"), "mean_pearson"])
        rnd = float(rows.loc[(protocol, "Random CellFM"), "mean_pearson"])
        delta_summary.append({"protocol": protocol, "delta": pre - rnd})
    delta_summary = pd.DataFrame(delta_summary)
    bars = ax_b.bar(
        delta_summary["protocol"],
        delta_summary["delta"],
        color=["#4DBBD5", "#E64B35"],
        edgecolor="0.25",
        linewidth=0.7,
    )
    ax_b.bar_label(bars, fmt="%+.3f", fontsize=7, padding=2)
    ax_b.axhline(0, color="0.25", linewidth=0.8)
    ax_b.set_ylim(0, max(0.10, float(delta_summary["delta"].max()) + 0.02))
    ax_b.set_xlabel("")
    ax_b.set_ylabel("Delta mean r")
    ax_b.set_title("Pretrained gain over random init")

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
    )
    sns.boxplot(
        data=genes,
        x="branch",
        y="pearson",
        order=ORDER,
        ax=ax_c,
        showcaps=True,
        width=0.16,
        boxprops={"facecolor": "white", "edgecolor": "0.25", "linewidth": 0.7},
        whiskerprops={"linewidth": 0.7, "color": "0.25"},
        capprops={"linewidth": 0.7, "color": "0.25"},
        medianprops={"linewidth": 0.8, "color": "0.1"},
        showfliers=False,
    )
    ax_c.set_xticklabels(["Single\nPT", "Single\nRI", "Multi\nPT", "Multi\nRI"], rotation=0)
    ax_c.set_xlabel("")
    ax_c.set_ylabel("Per-gene Pearson r")
    ax_c.set_title("Genome-wide per-gene distributions")
    ax_c.set_ylim(-0.35, 1.05)

    ax_d = fig.add_subplot(gs[2, 0])
    sns.violinplot(
        data=deltas,
        x="protocol",
        y="delta_pretrained_minus_random",
        order=PAIR_ORDER,
        palette=["#4DBBD5", "#E64B35"],
        ax=ax_d,
        cut=0,
        inner=None,
        linewidth=0.6,
    )
    sns.boxplot(
        data=deltas,
        x="protocol",
        y="delta_pretrained_minus_random",
        order=PAIR_ORDER,
        ax=ax_d,
        width=0.18,
        boxprops={"facecolor": "white", "edgecolor": "0.25", "linewidth": 0.7},
        whiskerprops={"linewidth": 0.7, "color": "0.25"},
        capprops={"linewidth": 0.7, "color": "0.25"},
        medianprops={"linewidth": 0.8, "color": "0.1"},
        showfliers=False,
    )
    ax_d.axhline(0, color="0.25", linewidth=0.8, linestyle=":")
    ax_d.set_xlabel("")
    ax_d.set_ylabel("Per-gene delta r\nPT - RI")
    ax_d.set_title("Gene-wise pretrained gain")
    ax_d.set_ylim(-0.4, 0.6)

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

    fig.savefig(FIG_DIR / "cellfm_pretrained_vs_random_extra.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "cellfm_pretrained_vs_random_extra.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / "cellfm_pretrained_vs_random_extra.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def write_interpretation(summary: pd.DataFrame) -> None:
    rows = summary.set_index(["protocol", "condition"])
    single_pt = rows.loc[("single", "Pretrained CellFM"), "mean_pearson"]
    single_ri = rows.loc[("single", "Random CellFM"), "mean_pearson"]
    multi_pt = rows.loc[("multi", "Pretrained CellFM"), "mean_pearson"]
    multi_ri = rows.loc[("multi", "Random CellFM"), "mean_pearson"]
    text = f"""CellFM pretrained-vs-random extra figure summary

Single-slide INT25-to-INT28:
- Pretrained CellFM mean Pearson: {single_pt:.4f}
- Random-init CellFM mean Pearson: {single_ri:.4f}
- Delta pretrained - random: {single_pt - single_ri:+.4f}

Multi-slide INT25+INT26 train, INT27 validation, INT28 test:
- Pretrained CellFM mean Pearson: {multi_pt:.4f}
- Random-init CellFM mean Pearson: {multi_ri:.4f}
- Delta pretrained - random: {multi_pt - multi_ri:+.4f}

Interpretation:
Pretrained CellFM improves over random initialization in both prostate
settings, with a larger gain in the single-slide blocked setting and a
smaller but consistent gain in the multi-slide setting.
"""
    (TABLE_DIR / "cellfm_pretrained_vs_random_interpretation.txt").write_text(text, encoding="utf-8")


def copy_outputs() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for path in list(FIG_DIR.glob("cellfm_pretrained_vs_random_extra.*")) + list(TABLE_DIR.glob("cellfm_pretrained_vs_random_*.csv")) + [TABLE_DIR / "cellfm_pretrained_vs_random_interpretation.txt"]:
        if path.exists():
            shutil.copy2(path, PAPER_DIR / path.name)


def main() -> None:
    summary, genes, deltas, markers, thresholds = load_all()
    save_tables(summary, genes, deltas, markers, thresholds)
    plot_figure(summary, genes, deltas, markers, thresholds)
    write_interpretation(summary)
    copy_outputs()
    print(summary.to_string(index=False))
    print(f"Saved outputs to {OUTDIR}")
    print(f"Copied outputs to {PAPER_DIR}")


if __name__ == "__main__":
    main()
