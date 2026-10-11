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
from PIL import Image
from skimage.transform import resize


ROOT = Path(r"E:\Morpho-FM")
FILTERED_ROOT = ROOT / "results" / "her2st_patient_h" / "whole_slide_fourier_notch"
SLIDES = ("H1", "H2", "H3")


def find_paper_dir() -> Path:
    desktop = Path.home() / "Desktop"
    for path in desktop.iterdir():
        if path.is_dir() and (path / "main.tex").exists() and (path / "Figure").exists():
            return path
    raise FileNotFoundError("Could not find manuscript directory under Desktop.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose compact HER2ST Fourier-notch dense marker summary panel.")
    parser.add_argument("--genes", default="KRT19,ERBB2,MGP,IGKC,CD74,ACTB")
    parser.add_argument("--max-display-dim", type=int, default=900)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def upsample_map(map_small: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    valid = np.isfinite(map_small).astype(np.float32)
    values = np.nan_to_num(map_small, nan=0.0).astype(np.float32)
    up_values = resize(values, target_hw, order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)
    up_valid = resize(valid, target_hw, order=0, preserve_range=True, anti_aliasing=False) > 0.5
    up_values[~up_valid] = np.nan
    return up_values


def normalize_panel(x: np.ndarray) -> np.ndarray:
    valid = np.isfinite(x)
    y = np.full_like(x, np.nan, dtype=np.float32)
    if not valid.any():
        return y
    vmin, vmax = np.percentile(x[valid], [1, 99])
    vmin = max(0.0, float(vmin))
    vmax = float(vmax)
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    y[valid] = np.clip((x[valid] - vmin) / (vmax - vmin), 0, 1)
    return y


def load_filtered(slide: str) -> tuple[np.ndarray, list[str]]:
    slide_dir = FILTERED_ROOT / f"train_{slide}_to_{slide}" / slide
    genes = json.loads((slide_dir / "pred_superres_selected_genes.json").read_text(encoding="utf-8"))
    maps = np.load(slide_dir / "pred_superres_selected_fourier_notch.npy", mmap_mode="r")
    return maps, genes


def load_display_he(slide: str, max_dim: int) -> np.ndarray:
    img = Image.open(ROOT / "data" / "superres_cache" / slide / "he.jpg").convert("RGB")
    scale = min(1.0, float(max_dim) / float(max(img.size)))
    if scale < 1.0:
        new_size = (max(1, int(round(img.size[0] * scale))), max(1, int(round(img.size[1] * scale))))
        img = img.resize(new_size, resample=Image.Resampling.BILINEAR)
    return np.array(img)


def main() -> None:
    args = parse_args()
    genes = [g.strip() for g in args.genes.split(",") if g.strip()]
    paper = find_paper_dir()
    out_dir = paper / "Figure" / "Figure_6" / "HER2ST_patient_H" / "12_dense_fourier_notch_marker_maps"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(SLIDES), len(genes), figsize=(6.5, 4.9))
    if len(SLIDES) == 1:
        axes = axes[None, :]
    last_im = None
    for row, slide in enumerate(SLIDES):
        he = load_display_he(slide, args.max_display_dim)
        maps, slide_genes = load_filtered(slide)
        for col, gene in enumerate(genes):
            ax = axes[row, col]
            ax.imshow(he)
            if gene in slide_genes:
                dense = np.asarray(maps[..., slide_genes.index(gene)], dtype=np.float32)
                up = upsample_map(dense, he.shape[:2])
                norm = normalize_panel(up)
                last_im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=1, alpha=np.where(np.isfinite(norm), 0.92, 0))
            ax.axis("off")
            if row == 0:
                ax.set_title(gene, fontsize=8, fontstyle="italic", pad=2)
            if col == 0:
                ax.text(
                    -0.04,
                    0.5,
                    slide,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                    fontsize=8,
                )
    cax = fig.add_axes([0.925, 0.18, 0.012, 0.64])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["low", "high"])
    fig.suptitle("HER2ST Fourier-notch dense marker maps", fontsize=9, y=0.985)
    fig.subplots_adjust(left=0.035, right=0.91, top=0.91, bottom=0.03, wspace=0.02, hspace=0.02)
    fig.savefig(out_dir / "HER2ST_fourier_notch_dense_marker_summary.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "HER2ST_fourier_notch_dense_marker_summary.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(out_dir / "HER2ST_fourier_notch_dense_marker_summary.png")


if __name__ == "__main__":
    main()
