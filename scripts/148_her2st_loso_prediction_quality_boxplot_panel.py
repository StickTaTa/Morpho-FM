from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GENES = ["ERBB2", "KRT19", "CD24", "MGP", "CD74", "IGKC"]
FOLDS = {
    "H1": ("loso_train_H2_H3_to_H1", "H2+H3"),
    "H2": ("loso_train_H1_H3_to_H2", "H1+H3"),
    "H3": ("loso_train_H1_H2_to_H3", "H1+H2"),
}
OUT_LOCAL = ROOT / "results" / "her2st_patient_h" / "leave_one_section_out_figures_high_pcc_markers" / "prediction_quality"
OUT_PAPER = (
    Path.home()
    / "Desktop"
    / "人生的第4篇论文"
    / "Figure"
    / "Figure_6"
    / "HER2ST_patient_H"
    / "16_loso_prediction_quality_summary"
)


def load_marker_pearson(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for section, (fold, train_label) in FOLDS.items():
        ckpt_dir = root / "checkpoints" / "her2st_patient_h" / "st_superres" / "leave_one_section_out" / fold
        result_dir = root / "results" / "her2st_patient_h" / "morpho" / "leave_one_section_out" / fold / section
        genes = json.loads((ckpt_dir / "common_genes.json").read_text(encoding="utf-8"))
        corr = np.load(result_dir / "gene_corr_all.npy")
        gene_to_corr = dict(zip(genes, corr))
        for gene in GENES:
            rows.append(
                {
                    "section": section,
                    "train_sections": train_label,
                    "gene": gene,
                    "pearson": float(gene_to_corr[gene]),
                }
            )
    return pd.DataFrame(rows)


def render_boxplot(root: Path, out_dir: Path, dpi: int = 2540) -> None:
    marker_corr = load_marker_pearson(root)
    width_in = 37.39 / 25.4
    height_in = 38.89 / 25.4
    fig, ax = plt.subplots(figsize=(width_in, height_in))

    box_data: list[np.ndarray] = []
    positions: list[float] = []
    colors: list[str] = []
    pos = 1.0
    for section, (fold, _) in FOLDS.items():
        result_dir = root / "results" / "her2st_patient_h" / "morpho" / "leave_one_section_out" / fold / section
        all_corr = np.load(result_dir / "gene_corr_all.npy")
        hvg_corr = np.load(result_dir / "gene_corr_hvg.npy")
        marker_vals = marker_corr[marker_corr["section"].eq(section)]["pearson"].to_numpy()
        for arr, color in [
            (all_corr[np.isfinite(all_corr)], "#b8d7f0"),
            (hvg_corr[np.isfinite(hvg_corr)], "#f2c36b"),
            (marker_vals[np.isfinite(marker_vals)], "#e98b8b"),
        ]:
            box_data.append(arr)
            positions.append(pos)
            colors.append(color)
            pos += 1.0
        pos += 0.7

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "0.15", "linewidth": 0.8},
        boxprops={"edgecolor": "0.25", "linewidth": 0.65},
        whiskerprops={"color": "0.35", "linewidth": 0.65},
        capprops={"color": "0.35", "linewidth": 0.65},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.95)

    rng = np.random.default_rng(7)
    for p, arr in zip(positions, box_data):
        sample = arr if arr.size <= 300 else rng.choice(arr, size=300, replace=False)
        ax.scatter(
            rng.normal(p, 0.035, size=sample.size),
            sample,
            s=1.8,
            c="0.25",
            alpha=0.16,
            linewidths=0,
            rasterized=True,
        )

    centers = [np.mean(positions[i * 3 : i * 3 + 3]) for i in range(3)]
    ax.set_xticks(centers)
    ax.set_xticklabels(["H1", "H2", "H3"], fontsize=7)
    ax.set_xlim(0.25, positions[-1] + 0.75)
    ax.set_ylim(-0.25, 1.0)
    ax.set_ylabel("Pearson r", fontsize=8)
    ax.tick_params(axis="y", labelsize=6.5, width=0.8, length=2.5)
    ax.tick_params(axis="x", width=0.8, length=2.5, pad=1)
    ax.grid(axis="y", color="0.88", linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    legend_handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color="#b8d7f0", label="All genes", markersize=3.8),
        plt.Line2D([0], [0], marker="s", linestyle="", color="#f2c36b", label="Top HVGs", markersize=3.8),
        plt.Line2D([0], [0], marker="s", linestyle="", color="#e98b8b", label="Selected markers", markersize=3.8),
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=5.8,
        frameon=False,
        loc="lower right",
        borderaxespad=0.25,
        handletextpad=0.3,
        labelspacing=0.25,
    )

    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.16, top=0.98)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "HER2ST_loso_prediction_quality_boxplot_37p39x38p89mm"
    for ext in ["pdf", "png", "tif"]:
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=dpi)
    marker_corr.to_csv(out_dir / "HER2ST_loso_selected_marker_pearson.csv", index=False)
    plt.close(fig)


def main() -> None:
    render_boxplot(ROOT, OUT_LOCAL)
    render_boxplot(ROOT, OUT_PAPER)
    print(OUT_LOCAL)
    print(OUT_PAPER)


if __name__ == "__main__":
    main()
