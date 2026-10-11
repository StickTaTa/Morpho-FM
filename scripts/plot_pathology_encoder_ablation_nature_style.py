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
TRAIN_ID = "INT25"
ABLATION_TAG = "resnet50_pca576_single_INT25_to_INT28"

RESULT_DIRS = {
    "HIPT pathology FM": ROOT / "results" / "prostate_hest" / "morpho" / "single_superres" / "INT25_to_INT28",
    "ImageNet ResNet50": ROOT
    / "results"
    / "ablation_pathology_encoder"
    / ABLATION_TAG
    / "INT28",
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

OUTDIR = ROOT / "figures_nature_encoder_ablation"
TABLE_DIR = OUTDIR / "tables"
FIG_DIR = OUTDIR / "figures"

ORDER = ["HIPT pathology FM", "ImageNet ResNet50"]
PALETTE = {
    "HIPT pathology FM": "#4DBBD5",
    "ImageNet ResNet50": "#E64B35",
}


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
        ROOT
        / "checkpoints"
        / "st_superres"
        / "pathology_encoder_ablation"
        / ABLATION_TAG
        / "common_genes.json",
        ROOT / "checkpoints" / "st_superres" / "cellfm_single_INT25_to_INT28" / "common_genes.json",
        ROOT / "checkpoints" / "st_superres" / "cellfm_forward_sum" / "common_genes.json",
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


def annotate_pval(ax: plt.Axes, idx1: int, idx2: int, y: float, h: float, pval: float) -> None:
    ax.plot([idx1, idx1, idx2, idx2], [y, y + h, y + h, y], lw=1.0, c="black")
    if pval < 0.001:
        star = "***"
    elif pval < 0.01:
        star = "**"
    elif pval < 0.05:
        star = "*"
    else:
        star = "ns"
    ax.text((idx1 + idx2) * 0.5, y + h + 0.02, f"({star})", ha="center", va="bottom", fontsize=8.5)


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    pairs: dict[str, dict[str, np.ndarray]] = {}
    ref_true = None
    n_genes = None

    for name, result_dir in RESULT_DIRS.items():
        pred, true = load_pair(result_dir)
        if n_genes is None:
            n_genes = pred.shape[1]
        elif pred.shape[1] != n_genes:
            raise ValueError(f"Gene count mismatch for {name}: {pred.shape[1]} vs {n_genes}")

        if ref_true is None:
            ref_true = true
        else:
            if true.shape != ref_true.shape:
                raise ValueError(f"True matrix shape mismatch for {name}: {true.shape} vs {ref_true.shape}")
            max_abs = float(np.max(np.abs(true - ref_true)))
            if max_abs > 1e-5:
                raise ValueError(f"True matrix differs for {name}; max abs difference={max_abs}")
        pairs[name] = {"pred": pred, "true": true}

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
    for group in ["All"] + BIN_LABELS:
        hipt = summary_df[(summary_df.model == "HIPT pathology FM") & (summary_df.group == group)][
            "mean_pearson"
        ].iloc[0]
        resnet = summary_df[(summary_df.model == "ImageNet ResNet50") & (summary_df.group == group)][
            "mean_pearson"
        ].iloc[0]
        delta_rows.append(
            {
                "comparison": "HIPT - ResNet50",
                "group": group,
                "delta_mean_pearson": float(hipt - resnet),
                "HIPT_mean": float(hipt),
                "ResNet50_mean": float(resnet),
            }
        )
    delta_df = pd.DataFrame(delta_rows)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    gene_df.to_csv(TABLE_DIR / "pathology_encoder_gene_level_pearson.csv", index=False, encoding="utf-8")
    summary_df.to_csv(TABLE_DIR / "pathology_encoder_summary_by_group.csv", index=False, encoding="utf-8")
    delta_df.to_csv(TABLE_DIR / "pathology_encoder_delta_HIPT_minus_ResNet50.csv", index=False, encoding="utf-8")

    marker_keep = [g for g in MARKER_GENES if g in genes]
    marker_rows = []
    for model_name in ORDER:
        corr = all_corrs[model_name]
        for gene in marker_keep:
            gi = genes.index(gene)
            marker_rows.append({"model": model_name, "gene": gene, "pearson": float(corr[gi])})
    marker_df = pd.DataFrame(marker_rows)
    marker_df.to_csv(TABLE_DIR / "pathology_encoder_marker_pearson.csv", index=False, encoding="utf-8")

    return gene_df, summary_df, marker_df, all_corrs


def plot_a1_violin(gene_df: pd.DataFrame) -> None:
    plot_df = gene_df[np.isfinite(gene_df["pearson"])].copy()
    plot_df["model"] = pd.Categorical(plot_df["model"], categories=ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(6.55, 3.6), dpi=300)
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
    ax.set_title(f"Pathology encoder ablation on {TEST_ID}: per-gene Pearson")
    ax.set_xlabel("")
    ax.set_ylabel("Pearson r")
    ax.set_ylim(-1.0, 1.25)
    ax.tick_params(axis="x", rotation=18)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")

    v1 = plot_df[plot_df["model"] == ORDER[0]]["pearson"].values
    v2 = plot_df[plot_df["model"] == ORDER[1]]["pearson"].values
    _, pval = stats.wilcoxon(v1, v2) if len(v1) == len(v2) else stats.ttest_ind(v1, v2, equal_var=False)
    annotate_pval(ax, 0, 1, 1.03, 0.05, float(pval))

    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    save_fig(fig, FIG_DIR, "A1_violin_pathology_encoder_per_gene_pearson")
    plt.close(fig)


def plot_a2_ecdf(all_corrs: dict[str, np.ndarray]) -> None:
    thresholds = np.linspace(-0.2, 1.0, 251)
    rows = []

    fig, ax = plt.subplots(figsize=(4.9, 3.15), dpi=300)
    for model_name in ORDER:
        corr = all_corrs[model_name]
        corr = corr[np.isfinite(corr)]
        fractions = np.array([(corr >= t).mean() for t in thresholds])
        rows.extend(
            {
                "model": model_name,
                "threshold": float(threshold),
                "fraction_genes_ge_threshold": float(frac),
            }
            for threshold, frac in zip(thresholds, fractions)
        )
        ax.plot(thresholds, fractions, color=PALETTE[model_name], linewidth=1.8, label=model_name)

    pd.DataFrame(rows).to_csv(TABLE_DIR / "pathology_encoder_ecdf_threshold_curve.csv", index=False)

    ax.set_title("Fraction of genes above Pearson threshold")
    ax.set_xlabel("Threshold (Pearson r)")
    ax.set_ylabel("Fraction of genes")
    ax.set_xlim(-0.2, 1.0)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="upper right")
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    save_fig(fig, FIG_DIR, "A2_ecdf_fraction_genes_pathology_encoder")
    plt.close(fig)


def plot_a3_expression_bins(summary_df: pd.DataFrame) -> None:
    bin_plot = summary_df[summary_df["group"].isin(BIN_LABELS)].copy()
    bin_plot["group"] = pd.Categorical(bin_plot["group"], categories=BIN_LABELS, ordered=True)
    bin_plot["model"] = pd.Categorical(bin_plot["model"], categories=ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(6.55, 3.25), dpi=300)
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
    ax.legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    save_fig(fig, FIG_DIR, "A3_bar_expression_bins_pathology_encoder")
    plt.close(fig)


def plot_s1_delta(summary_df: pd.DataFrame) -> None:
    rows = []
    for group in ["All"] + BIN_LABELS:
        hipt = summary_df[(summary_df.model == ORDER[0]) & (summary_df.group == group)]["mean_pearson"].iloc[0]
        resnet = summary_df[(summary_df.model == ORDER[1]) & (summary_df.group == group)]["mean_pearson"].iloc[0]
        rows.append({"group": group, "delta_mean_pearson": float(hipt - resnet)})
    delta_plot = pd.DataFrame(rows)
    delta_plot["group"] = pd.Categorical(delta_plot["group"], categories=["All"] + BIN_LABELS, ordered=True)

    fig, ax = plt.subplots(figsize=(4.9, 3.15), dpi=300)
    sns.pointplot(
        data=delta_plot,
        x="group",
        y="delta_mean_pearson",
        color="#3C5488",
        marker="o",
        linewidth=1.6,
        ax=ax,
    )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.65)
    ax.set_title("Encoder delta: HIPT - ResNet50")
    ax.set_xlabel("")
    ax.set_ylabel("Delta mean Pearson r")
    y_min = min(-0.02, float(delta_plot["delta_mean_pearson"].min()) - 0.006)
    y_max = max(0.02, float(delta_plot["delta_mean_pearson"].max()) + 0.006)
    ax.set_ylim(y_min, y_max)
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    save_fig(fig, FIG_DIR, "S1_delta_HIPT_minus_ResNet50_expression_bins")
    plt.close(fig)


def plot_b1_marker_heatmap(marker_df: pd.DataFrame) -> None:
    marker_keep = [g for g in MARKER_GENES if g in marker_df["gene"].unique()]
    pivot = marker_df.pivot(index="gene", columns="model", values="pearson").loc[marker_keep, ORDER]

    fig, ax = plt.subplots(figsize=(4.85, max(3.1, 0.34 * len(marker_keep))), dpi=300)
    sns.heatmap(
        pivot,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".3f",
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
    save_fig(fig, FIG_DIR, "B1_marker_heatmap_pathology_encoder")
    plt.close(fig)


def write_readme(summary_df: pd.DataFrame) -> None:
    all_rows = summary_df[summary_df["group"] == "All"].set_index("model")
    hipt = all_rows.loc["HIPT pathology FM"]
    resnet = all_rows.loc["ImageNet ResNet50"]
    delta = float(hipt["mean_pearson"] - resnet["mean_pearson"])
    readme = f"""# Pathology Encoder Ablation Nature-Style Figures

Setting: prostate single-slide blocked test, {TRAIN_ID} to {TEST_ID}.

Fixed components: pretrained CellFM backend, disk-shaped MIL bag, mean aggregation,
same gene set, and same spot-level target matrix.

Main comparison:

| Encoder | Mean Pearson | Median Pearson | n genes |
|---|---:|---:|---:|
| HIPT pathology FM | {float(hipt['mean_pearson']):.6f} | {float(hipt['median_pearson']):.6f} | {int(hipt['n_genes'])} |
| ImageNet ResNet50 | {float(resnet['mean_pearson']):.6f} | {float(resnet['median_pearson']):.6f} | {int(resnet['n_genes'])} |

Delta mean Pearson, HIPT minus ResNet50: {delta:+.6f}.

Recommended figure set:

- A1: per-gene Pearson violin plot.
- A2: Pearson-threshold ECDF curve.
- A3: true-expression-stratified mean Pearson bar plot.
- B1: fixed prostate marker-gene heatmap.
- S1: optional diagnostic delta plot.

Interpretation: this specific ablation does not support an independent performance
advantage of HIPT over the ImageNet ResNet50 encoder in the INT25-to-INT28 setting.
"""
    (OUTDIR / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    configure_style()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    gene_df, summary_df, marker_df, all_corrs = build_tables()
    plot_a1_violin(gene_df)
    plot_a2_ecdf(all_corrs)
    plot_a3_expression_bins(summary_df)
    plot_b1_marker_heatmap(marker_df)
    plot_s1_delta(summary_df)
    write_readme(summary_df)

    print(f"Saved figures: {FIG_DIR}")
    print(f"Saved tables: {TABLE_DIR}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
