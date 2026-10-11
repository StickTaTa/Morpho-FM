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

OUTDIR = ROOT / "figures_nature_encoder_ablation_single_multi"
FIG_DIR = OUTDIR / "figures"
TABLE_DIR = OUTDIR / "tables"

SINGLE_TAG = "resnet50_pca576_single_INT25_to_INT28"
MULTI_TAG = "resnet50_pca576_multi_INT25_INT26_val_INT27_to_INT28"

RUNS = [
    {
        "setting": "Single-slide",
        "train": "INT25",
        "val": "blocked INT25",
        "test": "INT28",
        "encoder": "HIPT pathology FM",
        "short_encoder": "HIPT",
        "result_dir": ROOT / "results" / "prostate_hest" / "morpho" / "single_superres" / "INT25_to_INT28",
        "gene_list": ROOT
        / "checkpoints"
        / "st_superres"
        / "pathology_encoder_ablation"
        / SINGLE_TAG
        / "common_genes.json",
    },
    {
        "setting": "Single-slide",
        "train": "INT25",
        "val": "blocked INT25",
        "test": "INT28",
        "encoder": "ImageNet ResNet50",
        "short_encoder": "ResNet50",
        "result_dir": ROOT / "results" / "ablation_pathology_encoder" / SINGLE_TAG / "INT28",
        "gene_list": ROOT
        / "checkpoints"
        / "st_superres"
        / "pathology_encoder_ablation"
        / SINGLE_TAG
        / "common_genes.json",
    },
    {
        "setting": "Multi-slide",
        "train": "INT25+INT26",
        "val": "INT27",
        "test": "INT28",
        "encoder": "HIPT pathology FM",
        "short_encoder": "HIPT",
        "result_dir": ROOT / "results" / "st_superres" / "cellfm_INT28",
        "gene_list": ROOT / "checkpoints" / "st_superres" / "cellfm_forward_sum" / "common_genes.json",
    },
    {
        "setting": "Multi-slide",
        "train": "INT25+INT26",
        "val": "INT27",
        "test": "INT28",
        "encoder": "ImageNet ResNet50",
        "short_encoder": "ResNet50",
        "result_dir": ROOT / "results" / "ablation_pathology_encoder" / MULTI_TAG / "INT28",
        "gene_list": ROOT
        / "checkpoints"
        / "st_superres"
        / "pathology_encoder_ablation"
        / MULTI_TAG
        / "common_genes.json",
    },
]

SETTING_ORDER = ["Single-slide", "Multi-slide"]
ENCODER_ORDER = ["HIPT pathology FM", "ImageNet ResNet50"]
MARKER_GENES = ["KLK3", "KLK2", "AR", "NKX3-1", "AMACR", "ERG", "EPCAM", "KRT8", "KRT18", "KRT5", "TP63"]
BIN_LABELS = ["Low", "Mid", "High"]

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
            "grid.linewidth": 0.55,
            "grid.alpha": 0.25,
            "axes.facecolor": "white",
        },
    )


def save_fig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")


def load_gene_list(path: Path, n_genes: int) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    genes = json.loads(path.read_text(encoding="utf-8"))
    genes = [str(g) for g in genes]
    if len(genes) != n_genes:
        raise ValueError(f"Gene-list length mismatch: {path}, {len(genes)} vs {n_genes}")
    return genes


def load_pair(result_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    pred_path = result_dir / "pred_bag.npy"
    true_path = result_dir / "true_bag.npy"
    if not pred_path.exists() or not true_path.exists():
        raise FileNotFoundError(f"Missing pred/true arrays in {result_dir}")
    pred = np.load(pred_path)
    true = np.load(true_path)
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch in {result_dir}: pred={pred.shape}, true={true.shape}")
    return pred, true


def pearson_corr_per_gene(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    true = true.astype(np.float64, copy=False)
    pred = pred.astype(np.float64, copy=False)
    t = true - true.mean(axis=0, keepdims=True)
    p = pred - pred.mean(axis=0, keepdims=True)
    den = np.sqrt((t * t).sum(axis=0) * (p * p).sum(axis=0))
    corr = np.full(true.shape[1], np.nan, dtype=np.float32)
    valid = den > 0
    corr[valid] = ((t[:, valid] * p[:, valid]).sum(axis=0) / den[valid]).astype(np.float32)
    return corr


def load_all_runs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    gene_rows = []
    summary_rows = []
    marker_rows = []
    delta_rows = []
    corr_map: dict[tuple[str, str], np.ndarray] = {}
    true_by_setting: dict[str, np.ndarray] = {}
    genes_by_setting: dict[str, list[str]] = {}

    for run in RUNS:
        pred, true = load_pair(run["result_dir"])
        genes = load_gene_list(run["gene_list"], pred.shape[1])
        setting = run["setting"]
        encoder = run["encoder"]

        if setting in true_by_setting:
            if true.shape != true_by_setting[setting].shape:
                raise ValueError(f"True matrix shape mismatch in {setting}")
            max_abs = float(np.max(np.abs(true - true_by_setting[setting])))
            if max_abs > 1e-5:
                raise ValueError(f"True matrix differs within {setting}; max abs={max_abs}")
            if genes != genes_by_setting[setting]:
                raise ValueError(f"Gene list differs within {setting}: {encoder}")
        else:
            true_by_setting[setting] = true
            genes_by_setting[setting] = genes

        corr = pearson_corr_per_gene(true, pred)
        corr_map[(setting, encoder)] = corr
        intensity = np.log1p(np.clip(true, 0, None)).mean(axis=0)
        q1, q2 = np.quantile(intensity, [1 / 3, 2 / 3])
        expr_bin = np.array(BIN_LABELS, dtype=object)[np.digitize(intensity, [q1, q2])]

        for idx, gene in enumerate(genes):
            gene_rows.append(
                {
                    "setting": setting,
                    "train": run["train"],
                    "val": run["val"],
                    "test": run["test"],
                    "encoder": encoder,
                    "short_encoder": run["short_encoder"],
                    "gene": gene,
                    "pearson": float(corr[idx]),
                    "true_expr_intensity": float(intensity[idx]),
                    "expr_bin": str(expr_bin[idx]),
                }
            )

        finite = np.isfinite(corr)
        summary_rows.append(
            {
                "setting": setting,
                "encoder": encoder,
                "group": "All",
                "n_spots": int(pred.shape[0]),
                "n_genes": int(finite.sum()),
                "mean_pearson": float(np.nanmean(corr)),
                "median_pearson": float(np.nanmedian(corr)),
            }
        )
        for group in BIN_LABELS:
            mask = expr_bin == group
            values = corr[mask]
            summary_rows.append(
                {
                    "setting": setting,
                    "encoder": encoder,
                    "group": group,
                    "n_spots": int(pred.shape[0]),
                    "n_genes": int(np.isfinite(values).sum()),
                    "mean_pearson": float(np.nanmean(values)),
                    "median_pearson": float(np.nanmedian(values)),
                }
            )

        for gene in MARKER_GENES:
            if gene in genes:
                gi = genes.index(gene)
                marker_rows.append(
                    {
                        "setting": setting,
                        "encoder": encoder,
                        "column": f"{setting}\n{run['short_encoder']}",
                        "gene": gene,
                        "pearson": float(corr[gi]),
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    for setting in SETTING_ORDER:
        for group in ["All"] + BIN_LABELS:
            hipt = summary_df[
                (summary_df.setting == setting)
                & (summary_df.encoder == "HIPT pathology FM")
                & (summary_df.group == group)
            ]["mean_pearson"].iloc[0]
            resnet = summary_df[
                (summary_df.setting == setting)
                & (summary_df.encoder == "ImageNet ResNet50")
                & (summary_df.group == group)
            ]["mean_pearson"].iloc[0]
            delta_rows.append(
                {
                    "setting": setting,
                    "group": group,
                    "delta_mean_pearson_HIPT_minus_ResNet50": float(hipt - resnet),
                    "HIPT_mean": float(hipt),
                    "ResNet50_mean": float(resnet),
                }
            )

    gene_df = pd.DataFrame(gene_rows)
    marker_df = pd.DataFrame(marker_rows)
    delta_df = pd.DataFrame(delta_rows)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    gene_df.to_csv(TABLE_DIR / "pathology_encoder_single_multi_gene_level_pearson.csv", index=False)
    summary_df.to_csv(TABLE_DIR / "pathology_encoder_single_multi_summary.csv", index=False)
    marker_df.to_csv(TABLE_DIR / "pathology_encoder_single_multi_marker_pearson.csv", index=False)
    delta_df.to_csv(TABLE_DIR / "pathology_encoder_single_multi_delta.csv", index=False)

    return gene_df, summary_df, marker_df, delta_df, corr_map


def plot_a1_violin(gene_df: pd.DataFrame) -> None:
    plot_df = gene_df[np.isfinite(gene_df["pearson"])].copy()
    plot_df["setting"] = pd.Categorical(plot_df["setting"], categories=SETTING_ORDER, ordered=True)
    plot_df["encoder"] = pd.Categorical(plot_df["encoder"], categories=ENCODER_ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(6.55, 3.65), dpi=300)
    sns.violinplot(
        data=plot_df,
        x="setting",
        y="pearson",
        hue="encoder",
        order=SETTING_ORDER,
        hue_order=ENCODER_ORDER,
        palette=[PALETTE[e] for e in ENCODER_ORDER],
        inner="box",
        cut=0,
        linewidth=0.85,
        ax=ax,
    )
    ax.set_title("Pathology encoder ablation across prostate splits")
    ax.set_xlabel("")
    ax.set_ylabel("Per-gene Pearson r")
    ax.set_ylim(-1.0, 1.15)
    ax.legend(frameon=False, loc="upper left", ncol=1)
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    save_fig(fig, "A1_violin_pathology_encoder_single_multi")
    plt.close(fig)


def plot_a2_ecdf(corr_map: dict[tuple[str, str], np.ndarray]) -> None:
    thresholds = np.linspace(-0.2, 1.0, 251)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(6.55, 3.05), dpi=300, sharey=True)
    for ax, setting in zip(axes, SETTING_ORDER):
        for encoder in ENCODER_ORDER:
            corr = corr_map[(setting, encoder)]
            corr = corr[np.isfinite(corr)]
            frac = np.array([(corr >= t).mean() for t in thresholds])
            rows.extend(
                {
                    "setting": setting,
                    "encoder": encoder,
                    "threshold": float(t),
                    "fraction_genes_ge_threshold": float(f),
                }
                for t, f in zip(thresholds, frac)
            )
            ax.plot(thresholds, frac, color=PALETTE[encoder], linewidth=1.7, label=encoder)
        ax.set_title(setting)
        ax.set_xlabel("Pearson threshold")
        ax.set_xlim(-0.2, 1.0)
        ax.set_ylim(0, 1.05)
        sns.despine(ax=ax, top=True, right=True)
    axes[0].set_ylabel("Fraction of genes")
    axes[1].legend(frameon=False, loc="upper right")
    pd.DataFrame(rows).to_csv(TABLE_DIR / "pathology_encoder_single_multi_ecdf.csv", index=False)
    plt.tight_layout()
    save_fig(fig, "A2_ecdf_pathology_encoder_single_multi")
    plt.close(fig)


def plot_a3_expression_bins(summary_df: pd.DataFrame) -> None:
    plot_df = summary_df[summary_df["group"].isin(BIN_LABELS)].copy()
    plot_df["setting"] = pd.Categorical(plot_df["setting"], categories=SETTING_ORDER, ordered=True)
    plot_df["group"] = pd.Categorical(plot_df["group"], categories=BIN_LABELS, ordered=True)
    fig, axes = plt.subplots(1, 2, figsize=(6.55, 3.2), dpi=300, sharey=True)
    for ax, setting in zip(axes, SETTING_ORDER):
        sns.barplot(
            data=plot_df[plot_df["setting"] == setting],
            x="group",
            y="mean_pearson",
            hue="encoder",
            hue_order=ENCODER_ORDER,
            palette=[PALETTE[e] for e in ENCODER_ORDER],
            ax=ax,
        )
        ax.set_title(setting)
        ax.set_xlabel("True-expression bin")
        ax.set_ylabel("Mean Pearson r" if ax is axes[0] else "")
        if ax is axes[1]:
            ax.legend(frameon=False, loc="upper right")
        else:
            ax.get_legend().remove()
        sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    save_fig(fig, "A3_expression_bins_pathology_encoder_single_multi")
    plt.close(fig)


def plot_a4_delta(delta_df: pd.DataFrame) -> None:
    plot_df = delta_df[delta_df["group"].isin(["All"] + BIN_LABELS)].copy()
    plot_df["group"] = pd.Categorical(plot_df["group"], categories=["All"] + BIN_LABELS, ordered=True)
    plot_df["setting"] = pd.Categorical(plot_df["setting"], categories=SETTING_ORDER, ordered=True)
    fig, ax = plt.subplots(figsize=(5.35, 3.25), dpi=300)
    sns.pointplot(
        data=plot_df,
        x="group",
        y="delta_mean_pearson_HIPT_minus_ResNet50",
        hue="setting",
        hue_order=SETTING_ORDER,
        palette=["#3C5488", "#00A087"],
        dodge=0.28,
        markers=["o", "s"],
        linewidth=1.5,
        ax=ax,
    )
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.65)
    ax.set_title("Encoder delta")
    ax.set_xlabel("")
    ax.set_ylabel("Mean Pearson delta\nHIPT - ResNet50")
    ax.legend(frameon=False, loc="best")
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    save_fig(fig, "A4_delta_HIPT_minus_ResNet50_single_multi")
    plt.close(fig)


def plot_b1_marker_heatmap(marker_df: pd.DataFrame) -> None:
    marker_keep = [gene for gene in MARKER_GENES if gene in marker_df["gene"].unique()]
    column_order = [
        "Single-slide\nHIPT",
        "Single-slide\nResNet50",
        "Multi-slide\nHIPT",
        "Multi-slide\nResNet50",
    ]
    pivot = marker_df.pivot(index="gene", columns="column", values="pearson").loc[marker_keep, column_order]
    fig, ax = plt.subplots(figsize=(6.25, max(3.25, 0.32 * len(marker_keep))), dpi=300)
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
    ax.set_title("Fixed prostate marker panel")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    sns.despine(ax=ax, left=True, bottom=True)
    plt.tight_layout()
    save_fig(fig, "B1_marker_heatmap_pathology_encoder_single_multi")
    plt.close(fig)


def write_manuscript_text(summary_df: pd.DataFrame, delta_df: pd.DataFrame) -> None:
    all_summary = summary_df[summary_df["group"] == "All"].copy()
    pivot = all_summary.pivot(index="setting", columns="encoder", values=["mean_pearson", "median_pearson"])
    single_hipt = float(pivot.loc["Single-slide", ("mean_pearson", "HIPT pathology FM")])
    single_resnet = float(pivot.loc["Single-slide", ("mean_pearson", "ImageNet ResNet50")])
    multi_hipt = float(pivot.loc["Multi-slide", ("mean_pearson", "HIPT pathology FM")])
    multi_resnet = float(pivot.loc["Multi-slide", ("mean_pearson", "ImageNet ResNet50")])
    single_delta = float(
        delta_df[(delta_df.setting == "Single-slide") & (delta_df.group == "All")][
            "delta_mean_pearson_HIPT_minus_ResNet50"
        ].iloc[0]
    )
    multi_delta = float(
        delta_df[(delta_df.setting == "Multi-slide") & (delta_df.group == "All")][
            "delta_mean_pearson_HIPT_minus_ResNet50"
        ].iloc[0]
    )
    text = f"""# Manuscript-Ready Text: Pathology Encoder Ablation

## Result Paragraph

To isolate the contribution of the pathology image encoder, we replaced the HIPT pathology foundation-model features with ImageNet-pretrained ResNet50 layer-3 features projected to the same feature dimensionality, while keeping the CellFM transcriptomic prior, disk-shaped spot bags, aggregation rule, gene set, and training split fixed. In the prostate single-slide setting (train INT25, blocked validation within INT25, test INT28), ResNet50 achieved a mean per-gene Pearson correlation of {single_resnet:.3f}, compared with {single_hipt:.3f} for HIPT (delta HIPT - ResNet50 = {single_delta:+.3f}). In the multi-slide setting (train INT25 and INT26, validate INT27, test INT28), the two encoders were again nearly indistinguishable, with mean Pearson correlations of {multi_resnet:.3f} for ResNet50 and {multi_hipt:.3f} for HIPT (delta = {multi_delta:+.3f}). These results indicate that, under the present prostate Visium evaluation, the performance gain of Morpho-FM is not driven by a clear independent advantage of the HIPT encoder over a strong ImageNet visual baseline; instead, the stronger evidence for the method remains the CellFM prior and the coupled morphology-to-transcriptome learning framework.

## Figure Caption

Pathology encoder ablation in prostate spatial transcriptomics. HIPT pathology foundation-model features were replaced by ImageNet-pretrained ResNet50 features while all downstream Morpho-FM components were kept fixed. (A) Per-gene Pearson correlations for HIPT and ResNet50 in single-slide and multi-slide prostate evaluation. (B) Fraction of genes exceeding each Pearson-correlation threshold. (C) Mean Pearson correlation stratified by true-expression level. (D) Mean-Pearson delta between HIPT and ResNet50; values above zero favor HIPT. (E) Pearson correlations for a fixed prostate marker-gene panel.

## Suggested Interpretation

Do not claim that this ablation proves HIPT is better than ImageNet ResNet50. The defensible statement is that HIPT and ResNet50 are comparable in this prostate test, with HIPT only slightly higher in the multi-slide split and lower in the single-slide split. If this figure is included in the main text, it should be framed as a specificity/control experiment rather than as positive evidence for a unique pathology-foundation-model contribution.
"""
    (OUTDIR / "pathology_encoder_single_multi_manuscript_text.md").write_text(text, encoding="utf-8")


def main() -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    gene_df, summary_df, marker_df, delta_df, corr_map = load_all_runs()
    plot_a1_violin(gene_df)
    plot_a2_ecdf(corr_map)
    plot_a3_expression_bins(summary_df)
    plot_a4_delta(delta_df)
    plot_b1_marker_heatmap(marker_df)
    write_manuscript_text(summary_df, delta_df)

    print(f"Saved figures to {FIG_DIR}")
    print(f"Saved tables to {TABLE_DIR}")
    print(summary_df.to_string(index=False))
    print(delta_df[delta_df.group == "All"].to_string(index=False))


if __name__ == "__main__":
    main()
