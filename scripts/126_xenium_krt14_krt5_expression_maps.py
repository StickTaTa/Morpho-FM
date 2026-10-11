from __future__ import annotations

import os
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


ROOT = Path(os.environ.get("MORPHO_VC_ROOT", r"E:\Morpho-FM")).resolve()
OUT_DIR = ROOT / "results" / "st_superres" / "xenium_gene_expression" / "KRT14_KRT5"
GENES = ["KRT14", "KRT5"]
SECTIONS = ["XEN1", "XEN2"]


def read_genes(cache_dir: Path) -> list[str]:
    return [line.strip() for line in (cache_dir / "gene-names.txt").read_text(encoding="utf-8").splitlines() if line.strip()]


def value_limits(values: np.ndarray) -> tuple[float, float]:
    pos = values[np.isfinite(values) & (values > 0)]
    if pos.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(pos, 1))
    vmax = float(np.percentile(pos, 99))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(pos.max())
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def load_section_gene(section: str, gene: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_dir = ROOT / "data" / "superres_cache" / section
    gene_names = read_genes(cache_dir)
    if gene not in gene_names:
        raise ValueError(f"{gene} not found in {cache_dir / 'gene-names.txt'}")

    locs = pd.read_csv(cache_dir / "locs.tsv", sep="\t")
    cnts = pd.read_csv(cache_dir / "cnts.tsv", sep="\t", usecols=[gene])
    values = cnts[gene].to_numpy(dtype=np.float32)
    coords = np.stack([locs["y"].to_numpy(dtype=np.float32), locs["x"].to_numpy(dtype=np.float32)], axis=1)

    Image.MAX_IMAGE_PIXELS = None
    he = np.array(Image.open(cache_dir / "he.jpg"))
    inside = (
        (coords[:, 1] >= 0)
        & (coords[:, 1] < he.shape[1])
        & (coords[:, 0] >= 0)
        & (coords[:, 0] < he.shape[0])
    )
    return coords[inside], values[inside], he


def plot_no_he(section: str, gene: str, coords: np.ndarray, counts: np.ndarray, he_shape: tuple[int, int], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_values = np.log1p(counts)
    positive = counts > 0
    vmin, vmax = value_limits(log_values)
    h, w = he_shape

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.scatter(
        coords[~positive, 1],
        coords[~positive, 0],
        s=2,
        color="#d9d9d9",
        alpha=0.16,
        edgecolors="none",
        rasterized=True,
    )
    sc = ax.scatter(
        coords[positive, 1],
        coords[positive, 0],
        s=2,
        c=log_values[positive],
        cmap="magma_r",
        vmin=vmin,
        vmax=vmax,
        alpha=0.92,
        edgecolors="none",
        rasterized=True,
    )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{section} {gene} measured expression", fontsize=10)
    ax.axis("off")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("log1p(count)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.05)
    pdf = out_dir / f"{section}_{gene}_measured_expression_no_he.pdf"
    png = out_dir / f"{section}_{gene}_measured_expression_no_he.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return png


def plot_he_overlay(section: str, gene: str, coords: np.ndarray, counts: np.ndarray, he: np.ndarray, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_values = np.log1p(counts)
    positive = counts > 0
    vmin, vmax = value_limits(log_values)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.imshow(he)
    sc = ax.scatter(
        coords[positive, 1],
        coords[positive, 0],
        s=2,
        c=log_values[positive],
        cmap="magma_r",
        vmin=vmin,
        vmax=vmax,
        alpha=0.88,
        edgecolors="none",
        rasterized=True,
    )
    ax.set_title(f"{section} {gene} measured expression on H&E", fontsize=10)
    ax.axis("off")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("log1p(count)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.05)
    pdf = out_dir / f"{section}_{gene}_measured_expression_he_overlay.pdf"
    png = out_dir / f"{section}_{gene}_measured_expression_he_overlay.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return png


def make_contact_sheet(image_paths: list[Path], out_path: Path, per_row: int = 2) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    widths, heights = zip(*(im.size for im in images))
    cell_w, cell_h = max(widths), max(heights)
    rows = (len(images) + per_row - 1) // per_row
    canvas = Image.new("RGB", (cell_w * per_row, cell_h * rows), color="white")
    for idx, im in enumerate(images):
        row, col = divmod(idx, per_row)
        x = col * cell_w + (cell_w - im.size[0]) // 2
        y = row * cell_h + (cell_h - im.size[1]) // 2
        canvas.paste(im, (x, y))
        im.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    no_he_paths: list[Path] = []
    he_paths: list[Path] = []
    rows = []
    for section in SECTIONS:
        for gene in GENES:
            coords, counts, he = load_section_gene(section, gene)
            no_he_png = plot_no_he(section, gene, coords, counts, he.shape[:2], OUT_DIR / "no_he_gray_reference")
            he_png = plot_he_overlay(section, gene, coords, counts, he, OUT_DIR / "he_overlay")
            no_he_paths.append(no_he_png)
            he_paths.append(he_png)
            rows.append(
                {
                    "section": section,
                    "gene": gene,
                    "n_cells_inside_he": int(len(counts)),
                    "n_positive_cells": int((counts > 0).sum()),
                    "positive_fraction": float((counts > 0).mean()),
                    "mean_raw_count": float(np.mean(counts)),
                    "max_raw_count": float(np.max(counts)),
                    "no_he_png": str(no_he_png),
                    "he_overlay_png": str(he_png),
                }
            )
    make_contact_sheet(no_he_paths, OUT_DIR / "KRT14_KRT5_measured_expression_no_he_contact_sheet.png")
    make_contact_sheet(he_paths, OUT_DIR / "KRT14_KRT5_measured_expression_he_overlay_contact_sheet.png")
    pd.DataFrame(rows).to_csv(OUT_DIR / "KRT14_KRT5_measured_expression_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"[done] {OUT_DIR}")


if __name__ == "__main__":
    main()
