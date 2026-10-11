from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(r"E:\Morpho-FM")
MANUSCRIPT_DIR = Path(r"C:\Users\Administrator\Desktop\人生的第4篇论文")
OUT_DIR = ROOT / "figures_nature_decoder_ablation_mlp_clean_rate"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
PAPER_DIR = MANUSCRIPT_DIR / "Figure" / "Supplementary_Figure" / "Decoder_ablation_mlp_clean_rate"

ROWS = [
    {
        "protocol": "Single",
        "setting": "INT25 -> INT28",
        "count_tag": "mlp_decoder_single_INT25_to_INT28",
        "clean_tag": "mlp_decoder_clean_rate_single_INT25_to_INT28",
        "color": "#4DBBD5",
    },
    {
        "protocol": "Multi",
        "setting": "INT25+INT26 -> INT28",
        "count_tag": "mlp_decoder_multi_INT25_INT26_val_INT27_to_INT28",
        "clean_tag": "mlp_decoder_clean_rate_multi_INT25_INT26_val_INT27_to_INT28",
        "color": "#E64B35",
    },
]


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
        }
    )


def load_metrics(path: Path) -> dict:
    return json.loads((path / "metrics.json").read_text(encoding="utf-8"))


def result_path(tag: str) -> Path:
    return ROOT / "results" / "ablation_decoder" / tag / "INT28"


def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    gene_rows = []
    for row in ROWS:
        count_dir = result_path(row["count_tag"])
        clean_dir = result_path(row["clean_tag"])
        count_metrics = load_metrics(count_dir)
        clean_metrics = load_metrics(clean_dir)

        for mode, metrics in [("Count-scale with measured SF", count_metrics), ("Clean rate-only", clean_metrics)]:
            metric_rows.append(
                {
                    "protocol": row["protocol"],
                    "setting": row["setting"],
                    "mode": mode,
                    "mean_pearson": metrics["Mean_Pearson"],
                    "median_pearson": metrics["Median_Pearson"],
                    "hvg1000_mean_pearson": metrics["HVG1000_Mean_Pearson"],
                    "hvg1000_median_pearson": metrics["HVG1000_Median_Pearson"],
                    "hvg1000_overlap": metrics["HVG1000_Overlap"],
                    "best_gene": metrics["Best_Gene"],
                    "best_pearson": metrics["Best_Pearson"],
                }
            )

        count_corr = np.load(count_dir / "gene_corr_all.npy")
        clean_corr = np.load(clean_dir / "gene_corr_all.npy")
        for corr, mode in [(count_corr, "Count-scale with measured SF"), (clean_corr, "Clean rate-only")]:
            finite = np.isfinite(corr)
            gene_rows.append(
                pd.DataFrame(
                    {
                        "protocol": row["protocol"],
                        "mode": mode,
                        "pearson": corr[finite],
                    }
                )
            )

    metrics_df = pd.DataFrame(metric_rows)
    gene_df = pd.concat(gene_rows, ignore_index=True)
    return metrics_df, gene_df


def save_tables(metrics_df: pd.DataFrame, gene_df: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(TABLE_DIR / "mlp_clean_rate_key_results.csv", index=False)
    summary = (
        gene_df.groupby(["protocol", "mode"])["pearson"]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .reset_index()
    )
    summary.to_csv(TABLE_DIR / "mlp_clean_rate_gene_distribution_summary.csv", index=False)


def plot_figure(metrics_df: pd.DataFrame, gene_df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    protocol_colors = {"Single": "#4DBBD5", "Multi": "#E64B35"}
    mode_colors = {"Count-scale with measured SF": "#9E9E9E", "Clean rate-only": "#3C5488"}

    fig = plt.figure(figsize=(6.65, 5.05), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], hspace=0.58, wspace=0.36)
    fig.subplots_adjust(top=0.88)

    ax_a = fig.add_subplot(gs[0, 0])
    x = np.arange(len(ROWS))
    width = 0.34
    for i, mode in enumerate(["Count-scale with measured SF", "Clean rate-only"]):
        sub = metrics_df[metrics_df["mode"] == mode].set_index("protocol")
        vals = [sub.loc[row["protocol"], "mean_pearson"] for row in ROWS]
        bars = ax_a.bar(
            x + (i - 0.5) * width,
            vals,
            width=width,
            color=mode_colors[mode],
            edgecolor="0.15",
            linewidth=0.5,
            label=mode,
        )
        ax_a.bar_label(bars, labels=[f"{v:.3f}" for v in vals], fontsize=6.6, padding=2)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([row["protocol"] for row in ROWS])
    ax_a.set_ylabel("Mean Pearson r")
    ax_a.set_title("MLP prediction protocol effect")
    ax_a.set_ylim(0, 0.43)

    ax_b = fig.add_subplot(gs[0, 1])
    for i, mode in enumerate(["Count-scale with measured SF", "Clean rate-only"]):
        sub = metrics_df[metrics_df["mode"] == mode].set_index("protocol")
        vals = [sub.loc[row["protocol"], "hvg1000_mean_pearson"] for row in ROWS]
        bars = ax_b.bar(
            x + (i - 0.5) * width,
            vals,
            width=width,
            color=mode_colors[mode],
            edgecolor="0.15",
            linewidth=0.5,
            label=mode,
        )
        ax_b.bar_label(bars, labels=[f"{v:.3f}" for v in vals], fontsize=6.6, padding=2)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([row["protocol"] for row in ROWS])
    ax_b.set_ylabel("HVG1000 mean Pearson r")
    ax_b.set_title("High-variance gene panel")
    ax_b.set_ylim(0, 0.95)

    fig.legend(
        handles=[
            Patch(facecolor=mode_colors["Count-scale with measured SF"], edgecolor="0.15", label="Count-scale with measured test SF"),
            Patch(facecolor=mode_colors["Clean rate-only"], edgecolor="0.15", label="Clean rate-only, no test SF in prediction"),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        handlelength=1.2,
        columnspacing=1.8,
    )

    ax_c = fig.add_subplot(gs[1, 0])
    positions = [0.8, 1.2, 1.8, 2.2]
    data = []
    colors = []
    for protocol in ["Single", "Multi"]:
        for mode in ["Count-scale with measured SF", "Clean rate-only"]:
            values = gene_df[(gene_df["protocol"] == protocol) & (gene_df["mode"] == mode)]["pearson"].to_numpy()
            data.append(values[np.isfinite(values)])
            colors.append(mode_colors[mode])
    parts = ax_c.violinplot(data, positions=positions, widths=0.28, showmeans=False, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("0.15")
        body.set_alpha(0.75)
        body.set_linewidth(0.5)
    parts["cmedians"].set_color("0.1")
    parts["cmedians"].set_linewidth(0.8)
    ax_c.axhline(0, color="0.3", linewidth=0.7, linestyle=":")
    ax_c.set_xticks([1.0, 2.0])
    ax_c.set_xticklabels(["Single", "Multi"])
    ax_c.set_ylabel("Per-gene Pearson r")
    ax_c.set_title("Genome-wide gene distributions")
    ax_c.set_ylim(-0.25, 1.02)

    ax_d = fig.add_subplot(gs[1, 1])
    for row in ROWS:
        count_corr = np.load(result_path(row["count_tag"]) / "gene_corr_all.npy")
        clean_corr = np.load(result_path(row["clean_tag"]) / "gene_corr_all.npy")
        finite = np.isfinite(count_corr) & np.isfinite(clean_corr)
        idx = np.where(finite)[0]
        if idx.size > 5000:
            rng = np.random.default_rng(42)
            idx = rng.choice(idx, size=5000, replace=False)
        ax_d.scatter(
            count_corr[idx],
            clean_corr[idx],
            s=3,
            alpha=0.18,
            color=protocol_colors[row["protocol"]],
            linewidths=0,
            label=row["protocol"],
        )
    ax_d.axhline(0, color="0.3", linewidth=0.7, linestyle=":")
    ax_d.axvline(0, color="0.3", linewidth=0.7, linestyle=":")
    ax_d.set_xlim(-0.15, 1.02)
    ax_d.set_ylim(-0.25, 0.65)
    ax_d.set_xlabel("Count-scale Pearson r\nwith measured test SF")
    ax_d.set_ylabel("Clean rate-only Pearson r")
    ax_d.set_title("Per-gene agreement shift")
    ax_d.legend(frameon=False, loc="upper left")

    for label, ax in zip(["a", "b", "c", "d"], [ax_a, ax_b, ax_c, ax_d]):
        ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig(FIG_DIR / "mlp_clean_rate_diagnostic.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "mlp_clean_rate_diagnostic.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / "mlp_clean_rate_diagnostic.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def copy_to_manuscript() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for path in list(FIG_DIR.glob("mlp_clean_rate_diagnostic.*")) + list(TABLE_DIR.glob("*.csv")):
        shutil.copy2(path, PAPER_DIR / path.name)


def main() -> None:
    configure_matplotlib()
    metrics_df, gene_df = load_all()
    save_tables(metrics_df, gene_df)
    plot_figure(metrics_df, gene_df)
    copy_to_manuscript()
    print(metrics_df.to_string(index=False))
    print(f"Saved figures to {FIG_DIR}")
    print(f"Copied outputs to {PAPER_DIR}")


if __name__ == "__main__":
    main()
