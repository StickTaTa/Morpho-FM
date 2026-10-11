from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "st_superres" / "xenium_extended"
PANEL_DIR = OUT_DIR / "panel_cpu"
FIG_DIR = PANEL_DIR / "figures"
PANEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

PANEL_BY_CATEGORY: dict[str, list[str]] = {
    "tumour_luminal": ["ERBB2", "ESR1", "PGR", "FOXA1", "GATA3"],
    "epithelial": ["EPCAM", "KRT7", "KRT8", "TACSTD2", "CEACAM6"],
    "proliferation": ["MKI67", "TOP2A", "CCND1", "CENPF"],
    "stromal": ["LUM", "POSTN", "MMP2", "LRRC15", "SFRP1"],
    "vascular": ["PECAM1", "VWF", "RAMP2", "AQP1", "CAV1"],
    "immune": ["PTPRC", "CD3D", "CD4", "CD8A", "C1QA"],
}

RUNS = [
    {
        "run_name": "XEN1_within",
        "setting": "XEN1 train -> XEN1 predict",
        "section": "XEN1",
        "train_section": "XEN1",
        "mode": "within",
        "result_dir": ROOT / "results" / "st_superres" / "xenium_cellfm_1" / "xenium_XEN1",
    },
    {
        "run_name": "XEN2_to_XEN1",
        "setting": "XEN2 train -> XEN1 predict",
        "section": "XEN1",
        "train_section": "XEN2",
        "mode": "cross",
        "result_dir": ROOT / "results" / "st_superres" / "xenium_cellfm_2" / "xenium_XEN1",
    },
    {
        "run_name": "XEN2_within",
        "setting": "XEN2 train -> XEN2 predict",
        "section": "XEN2",
        "train_section": "XEN2",
        "mode": "within",
        "result_dir": ROOT / "results" / "st_superres" / "xenium_cellfm_2" / "xenium_XEN2",
    },
    {
        "run_name": "XEN1_to_XEN2",
        "setting": "XEN1 train -> XEN2 predict",
        "section": "XEN2",
        "train_section": "XEN1",
        "mode": "cross",
        "result_dir": ROOT / "results" / "st_superres" / "xenium_cellfm_1" / "xenium_XEN2",
    },
]


def panel_records() -> pd.DataFrame:
    rows = []
    rank = 0
    for category, genes in PANEL_BY_CATEGORY.items():
        for gene in genes:
            rows.append({"gene": gene, "category": category, "panel_rank": rank})
            rank += 1
    return pd.DataFrame(rows)


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "q25": np.nan, "median": np.nan, "q75": np.nan, "max": np.nan}
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "max": float(values.max()),
    }


def compute_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common_genes = json.loads((ROOT / "checkpoints" / "st_superres" / "xenium_cellfm" / "common_genes.json").read_text(encoding="utf-8"))
    gene_to_idx = {gene: idx for idx, gene in enumerate(common_genes)}
    panel_df = panel_records()
    missing = sorted(set(panel_df["gene"]) - set(gene_to_idx))
    if missing:
        raise ValueError(f"Panel genes missing from common genes: {missing}")

    full_rows = []
    panel_rows = []
    for run in RUNS:
        corrs = np.load(run["result_dir"] / "gene_corrs.npy")
        for gene, corr in zip(common_genes, corrs):
            full_rows.append({**{k: run[k] for k in ["run_name", "setting", "section", "train_section", "mode"]}, "gene": gene, "spot_pearson": float(corr)})
        for _, rec in panel_df.iterrows():
            gene = rec["gene"]
            panel_rows.append(
                {
                    **{k: run[k] for k in ["run_name", "setting", "section", "train_section", "mode"]},
                    "gene": gene,
                    "category": rec["category"],
                    "panel_rank": int(rec["panel_rank"]),
                    "spot_pearson": float(corrs[gene_to_idx[gene]]),
                }
            )

    full_df = pd.DataFrame(full_rows)
    panel_metrics = pd.DataFrame(panel_rows)

    summary_rows = []
    for run in RUNS:
        full_vals = full_df.loc[full_df["run_name"] == run["run_name"], "spot_pearson"].to_numpy()
        panel_vals = panel_metrics.loc[panel_metrics["run_name"] == run["run_name"], "spot_pearson"].to_numpy()
        for metric, values in [("full_306_spot_pearson", full_vals), ("balanced_29_spot_pearson", panel_vals)]:
            row = {k: run[k] for k in ["run_name", "setting", "section", "train_section", "mode"]}
            row["metric"] = metric
            row.update(summarize(values))
            summary_rows.append(row)

    category_rows = []
    for (run_name, category), df in panel_metrics.groupby(["run_name", "category"], sort=False):
        run = next(item for item in RUNS if item["run_name"] == run_name)
        row = {k: run[k] for k in ["run_name", "setting", "section", "train_section", "mode"]}
        row["category"] = category
        row["metric"] = "balanced_29_spot_pearson"
        row.update(summarize(df["spot_pearson"].to_numpy()))
        category_rows.append(row)

    return panel_df, full_df, panel_metrics, pd.DataFrame(summary_rows), pd.DataFrame(category_rows)


def plot_distributions(full_df: pd.DataFrame, panel_metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    specs = [
        ("306-gene spot Pearson", full_df, "spot_pearson"),
        ("Balanced 29-gene spot Pearson", panel_metrics, "spot_pearson"),
    ]
    colors = {"within": "#2878b5", "cross": "#c75d2c"}
    for ax, (title, df, col) in zip(axes, specs):
        data = [df.loc[df["run_name"] == run["run_name"], col].dropna().to_numpy() for run in RUNS]
        vp = ax.violinplot(data, showmedians=True, widths=0.78)
        for body, run in zip(vp["bodies"], RUNS):
            body.set_facecolor(colors[run["mode"]])
            body.set_edgecolor("black")
            body.set_alpha(0.65)
        for key in ["cbars", "cmins", "cmaxes", "cmedians"]:
            vp[key].set_color("black")
        ax.boxplot(data, widths=0.18, showfliers=False, patch_artist=True, boxprops={"facecolor": "white", "alpha": 0.75})
        ax.set_title(title)
        ax.set_xticks(range(1, len(RUNS) + 1))
        ax.set_xticklabels([run["setting"].replace(" -> ", "\n-> ") for run in RUNS], fontsize=8)
        ax.set_ylim(-0.05, 1.0)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Pearson correlation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "xenium_spot_distribution_four_settings.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "xenium_spot_distribution_four_settings.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_category_heatmap(category_summary: pd.DataFrame) -> None:
    categories = list(PANEL_BY_CATEGORY.keys())
    pivot = category_summary.pivot(index="category", columns="setting", values="mean")
    pivot = pivot.loc[categories, [run["setting"] for run in RUNS]]
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    im = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([col.replace(" -> ", "\n-> ") for col in pivot.columns], fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.55 else "black", fontsize=8)
    ax.set_title("Balanced 29-gene spot Pearson by category")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Mean Pearson")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "xenium_balanced_panel_category_spot_heatmap.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "xenium_balanced_panel_category_spot_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    panel_df, full_df, panel_metrics, summary_df, category_summary = compute_metrics()
    panel_df.to_csv(PANEL_DIR / "xenium_balanced_panel_genes.csv", index=False)
    full_df.to_csv(PANEL_DIR / "xenium_306gene_spot_metrics_detail.csv", index=False)
    panel_metrics.to_csv(PANEL_DIR / "xenium_balanced_panel_spot_metrics_detail.csv", index=False)
    summary_df.to_csv(PANEL_DIR / "xenium_spot_distribution_summary.csv", index=False)
    category_summary.to_csv(PANEL_DIR / "xenium_balanced_panel_category_spot_summary.csv", index=False)

    avg = (
        panel_metrics.groupby(["gene", "category", "mode"], as_index=False)["spot_pearson"]
        .mean()
        .pivot(index=["gene", "category"], columns="mode", values="spot_pearson")
        .reset_index()
    )
    avg.to_csv(PANEL_DIR / "xenium_balanced_panel_spot_average_by_mode.csv", index=False)
    plot_distributions(full_df, panel_metrics)
    plot_category_heatmap(category_summary)
    print("[done] CPU-only Xenium panel summaries written to", PANEL_DIR)


if __name__ == "__main__":
    main()
