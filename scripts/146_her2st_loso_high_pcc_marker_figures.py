from __future__ import annotations

import argparse
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
from PIL import Image
from skimage.transform import resize


ROOT = Path(__file__).resolve().parents[1]
GENES = ["ERBB2", "KRT19", "CD24", "MGP", "CD74", "IGKC"]
FOLDS = {
    "H1": ("loso_train_H2_H3_to_H1", "H2+H3"),
    "H2": ("loso_train_H1_H3_to_H2", "H1+H3"),
    "H3": ("loso_train_H1_H2_to_H3", "H1+H2"),
}
SECTION_FOLDERS = {
    "H1": "16_loso_H1_transfer_summary",
    "H2": "16_loso_H2_transfer_summary",
    "H3": "16_loso_H3_transfer_summary",
}

Image.MAX_IMAGE_PIXELS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render HER2ST LOSO high-PCC marker figures.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--dense-root",
        type=Path,
        default=ROOT / "results" / "her2st_patient_h" / "whole_slide_loso_high_pcc_markers",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "results" / "her2st_patient_h" / "leave_one_section_out_figures_high_pcc_markers",
    )
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=Path.home()
        / "Desktop"
        / "人生的第4篇论文"
        / "Figure"
        / "Figure_6"
        / "HER2ST_patient_H",
    )
    parser.add_argument("--genes", default=",".join(GENES))
    parser.add_argument("--dpi", type=int, default=900)
    return parser.parse_args()


def single_vrange(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(arr, [1, 99])
    vmin = max(0.0, float(vmin))
    vmax = float(vmax)
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def load_display_he(path: Path, max_dim: int = 4800) -> tuple[np.ndarray, float, float]:
    img = Image.open(path)
    orig_w, orig_h = img.size
    scale = min(1.0, float(max_dim) / float(max(orig_w, orig_h)))
    if scale < 1.0:
        img = img.resize(
            (int(round(orig_w * scale)), int(round(orig_h * scale))),
            resample=Image.Resampling.LANCZOS,
        )
    arr = np.array(img)
    return arr, arr.shape[1] / float(orig_w), arr.shape[0] / float(orig_h)


def upsample_map(map_small: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    valid = np.isfinite(map_small).astype(np.float32)
    values = np.nan_to_num(map_small, nan=0.0).astype(np.float32)
    up_values = resize(values, target_hw, order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)
    up_valid = resize(valid, target_hw, order=0, preserve_range=True, anti_aliasing=False) > 0.5
    up_values[~up_valid] = np.nan
    return up_values


def load_measured(root: Path, section: str, genes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    cnts = pd.read_csv(root / "data" / "superres_cache" / section / "cnts.tsv", sep="\t", index_col=0)
    locs = pd.read_csv(root / "data" / "superres_cache" / section / "locs.tsv", sep="\t", index_col=0)
    common = cnts.index.intersection(locs.index)
    cnts = cnts.loc[common, genes]
    locs = locs.loc[common]
    coords = locs[["x", "y"]].to_numpy(dtype=np.float32)
    return cnts.to_numpy(dtype=np.float32), coords


def load_marker_pearson(root: Path, genes: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for section, (fold, train_label) in FOLDS.items():
        ckpt_dir = root / "checkpoints" / "her2st_patient_h" / "st_superres" / "leave_one_section_out" / fold
        result_dir = root / "results" / "her2st_patient_h" / "morpho" / "leave_one_section_out" / fold / section
        all_genes = json.loads((ckpt_dir / "common_genes.json").read_text(encoding="utf-8"))
        corr = np.load(result_dir / "gene_corr_all.npy")
        gene_to_corr = dict(zip(all_genes, corr))
        for gene in genes:
            rows.append(
                {
                    "section": section,
                    "train_sections": train_label,
                    "target_section": section,
                    "gene": gene,
                    "pearson": float(gene_to_corr[gene]),
                }
            )
    return pd.DataFrame(rows)


def render_marker_section(
    root: Path,
    dense_root: Path,
    out_dir: Path,
    section: str,
    fold: str,
    train_label: str,
    genes: list[str],
    marker_corr: pd.DataFrame,
    dpi: int,
) -> None:
    dense_dir = dense_root / fold / section
    pred_maps = np.load(dense_dir / "pred_superres_selected.npy")
    saved_genes = json.loads((dense_dir / "pred_superres_selected_genes.json").read_text(encoding="utf-8"))
    idx = [saved_genes.index(g) for g in genes]
    measured, coords = load_measured(root, section, genes)
    he, sx, sy = load_display_he(root / "data" / "superres_cache" / section / "he.jpg")
    coords = coords.copy()
    coords[:, 0] *= sx
    coords[:, 1] *= sy
    corr_map = marker_corr[marker_corr["section"].eq(section)].set_index("gene")["pearson"].to_dict()

    fig, axes = plt.subplots(len(genes), 2, figsize=(6.69, 8.25))
    for row, (gene, gi, mi) in enumerate(zip(genes, idx, range(len(genes)))):
        dense_up = upsample_map(np.asarray(pred_maps[..., gi], dtype=np.float32), he.shape[:2])
        mvmin, mvmax = single_vrange(measured[:, mi])
        dvmin, dvmax = single_vrange(dense_up)

        ax = axes[row, 0]
        ax.imshow(he)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=measured[:, mi],
            s=7,
            cmap="viridis",
            vmin=mvmin,
            vmax=mvmax,
            alpha=0.86,
            edgecolors="none",
            rasterized=True,
        )
        ax.set_ylabel(gene, fontsize=8, rotation=0, labelpad=22, va="center", fontstyle="italic")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Measured spots" if row == 0 else "", fontsize=9, pad=2)

        ax = axes[row, 1]
        ax.imshow(he)
        alpha = np.where(np.isfinite(dense_up), 0.93, 0.0)
        ax.imshow(dense_up, cmap="viridis", vmin=dvmin, vmax=dvmax, alpha=alpha)
        ax.text(
            0.02,
            0.96,
            f"r={corr_map[gene]:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.0},
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Dense held-out prediction" if row == 0 else "", fontsize=9, pad=2)

    cax = fig.add_axes([0.91, 0.16, 0.012, 0.68])
    grad = np.linspace(0, 1, 256)[:, None]
    cax.imshow(grad, cmap="viridis", origin="lower", aspect="auto")
    cax.set_xticks([])
    cax.set_yticks([0, 255])
    cax.set_yticklabels(["low", "high"], fontsize=7)
    cax.set_ylabel("Relative expression", fontsize=8)
    fig.suptitle(f"{section} held-out marker reconstruction (train {train_label})", fontsize=10, y=0.995)
    fig.subplots_adjust(left=0.08, right=0.89, top=0.96, bottom=0.02, wspace=0.03, hspace=0.04)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{section}_loso_high_pcc_marker_summary"
    fig.savefig(out_dir / f"{stem}.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.tif", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_prediction_quality(root: Path, out_dir: Path, genes: list[str], marker_corr: pd.DataFrame, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.69, 2.75), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    box_data = []
    positions = []
    colors = []
    labels = []
    pos = 1
    for section, (fold, _) in FOLDS.items():
        result_dir = root / "results" / "her2st_patient_h" / "morpho" / "leave_one_section_out" / fold / section
        all_corr = np.load(result_dir / "gene_corr_all.npy")
        hvg_corr = np.load(result_dir / "gene_corr_hvg.npy")
        marker_vals = marker_corr[marker_corr["section"].eq(section)]["pearson"].to_numpy()
        for arr, color, label in [
            (all_corr[np.isfinite(all_corr)], "#b8d7f0", "All genes"),
            (hvg_corr[np.isfinite(hvg_corr)], "#f2c36b", "Top HVGs"),
            (marker_vals[np.isfinite(marker_vals)], "#e98b8b", "Selected markers"),
        ]:
            box_data.append(arr)
            positions.append(pos)
            colors.append(color)
            labels.append(label)
            pos += 1
        pos += 0.7
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "0.15", "linewidth": 0.9},
        boxprops={"edgecolor": "0.25", "linewidth": 0.7},
        whiskerprops={"color": "0.35", "linewidth": 0.7},
        capprops={"color": "0.35", "linewidth": 0.7},
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
            s=3,
            c="0.25",
            alpha=0.18,
            linewidths=0,
            rasterized=True,
        )
    centers = [np.mean(positions[i * 3 : i * 3 + 3]) for i in range(3)]
    ax.set_xticks(centers)
    ax.set_xticklabels(["H1", "H2", "H3"], fontsize=8)
    ax.set_ylim(-0.25, 1.0)
    ax.set_ylabel("Pearson r", fontsize=8)
    ax.set_title("Held-out prediction", fontsize=9, pad=3)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    legend_handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color="#b8d7f0", label="All genes", markersize=6),
        plt.Line2D([0], [0], marker="s", linestyle="", color="#f2c36b", label="Top HVGs", markersize=6),
        plt.Line2D([0], [0], marker="s", linestyle="", color="#e98b8b", label="Selected markers", markersize=6),
    ]
    ax.legend(handles=legend_handles, fontsize=7, frameon=False, loc="lower right")

    ax = axes[1]
    heat = marker_corr.pivot(index="gene", columns="section", values="pearson").loc[genes, ["H1", "H2", "H3"]]
    im = ax.imshow(heat.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["H1", "H2", "H3"], fontsize=8)
    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels(genes, fontsize=8, fontstyle="italic")
    ax.set_title("Selected marker r", fontsize=9, pad=3)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color="white" if val < 0.55 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Pearson r", fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.15, wspace=0.35)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "HER2ST_loso_prediction_quality_summary"
    fig.savefig(out_dir / f"{stem}.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.tif", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    genes = [g.strip() for g in args.genes.split(",") if g.strip()]
    marker_corr = load_marker_pearson(root, genes)
    args.out_root.mkdir(parents=True, exist_ok=True)
    marker_corr.to_csv(args.out_root / "HER2ST_loso_selected_marker_pearson.csv", index=False)

    for section, (fold, train_label) in FOLDS.items():
        local_out = args.out_root / section / fold
        render_marker_section(root, args.dense_root, local_out, section, fold, train_label, genes, marker_corr, args.dpi)
        paper_out = args.paper_root / SECTION_FOLDERS[section]
        paper_out.mkdir(parents=True, exist_ok=True)
        for ext in ["pdf", "png", "tif"]:
            src = local_out / f"{section}_loso_high_pcc_marker_summary.{ext}"
            dst = paper_out / src.name
            dst.write_bytes(src.read_bytes())
        marker_corr[marker_corr["section"].eq(section)].to_csv(
            paper_out / f"{section}_loso_high_pcc_marker_pearson.csv",
            index=False,
        )

    quality_local = args.out_root / "prediction_quality"
    render_prediction_quality(root, quality_local, genes, marker_corr, args.dpi)
    quality_paper = args.paper_root / "16_loso_prediction_quality_summary"
    quality_paper.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png", "tif"]:
        src = quality_local / f"HER2ST_loso_prediction_quality_summary.{ext}"
        (quality_paper / src.name).write_bytes(src.read_bytes())
    marker_corr.to_csv(quality_paper / "HER2ST_loso_selected_marker_pearson.csv", index=False)
    print(args.out_root)
    print(quality_paper)


if __name__ == "__main__":
    main()
