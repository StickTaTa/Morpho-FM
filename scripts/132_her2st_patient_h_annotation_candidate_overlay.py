from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from PIL import Image
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree


ROOT = Path(r"E:\Morpho-FM")
OUT_DIR = Path(
    r"C:\Users\Administrator\Desktop\人生的第4篇论文\Figure\Figure_6"
    r"\HER2ST_patient_H\04_annotation_candidate_domains"
)

SLIDES = ("H1", "H2", "H3")

LABEL_COLORS = {
    "invasive cancer": "#e41a1c",
    "cancer in situ": "#ff7f00",
    "immune infiltrate": "#f2df00",
    "breast glands": "#1ac938",
    "adipose tissue": "#52d1c6",
    "connective tissue": "#3b4cc0",
}

DOMAIN_COLORS = {
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#2ca02c",
    3: "#d62728",
    4: "#9467bd",
    5: "#8c564b",
    6: "#17becf",
}


def read_image(slide: str) -> Image.Image:
    return Image.open(
        ROOT / "data" / "HER2ST_breast_cancer" / "raw" / "images" / "HE" / f"{slide}.jpg"
    ).convert("RGB")


def read_annotation_image() -> Image.Image:
    return Image.open(
        ROOT / "data" / "HER2ST_breast_cancer" / "raw" / "images" / "annotation" / "H1.jpg"
    ).convert("RGB")


def read_pred_clusters(slide: str) -> pd.DataFrame:
    path = (
        ROOT
        / "results"
        / "her2st_patient_h"
        / "clustering"
        / f"pred_bag_train_{slide}_to_{slide}"
        / slide
        / "cluster_assignments.csv"
    )
    return pd.read_csv(path)


def read_h1_labels() -> pd.DataFrame:
    path = (
        ROOT
        / "data"
        / "HER2ST_breast_cancer"
        / "raw"
        / "meta"
        / "H1_labeled_coordinates.tsv"
    )
    return pd.read_csv(path, sep="\t")


def panel_bounds(points: pd.DataFrame, image: Image.Image, margin: int = 450) -> tuple[int, int, int, int]:
    x0 = max(0, int(points["pixel_x"].min()) - margin)
    x1 = min(image.size[0], int(points["pixel_x"].max()) + margin)
    y0 = max(0, int(points["pixel_y"].min()) - margin)
    y1 = min(image.size[1], int(points["pixel_y"].max()) + margin)
    return x0, x1, y0, y1


def crop_nonwhite(image: Image.Image, threshold: int = 246, margin: int = 60) -> Image.Image:
    arr = np.asarray(image)
    nonwhite = np.any(arr < threshold, axis=2)
    ys, xs = np.where(nonwhite)
    if len(xs) == 0:
        return image
    x0 = max(0, xs.min() - margin)
    x1 = min(image.size[0], xs.max() + margin)
    y0 = max(0, ys.min() - margin)
    y1 = min(image.size[1], ys.max() + margin)
    return image.crop((x0, y0, x1, y1))


def draw_background(ax, image: Image.Image, bounds: tuple[int, int, int, int]) -> None:
    x0, x1, y0, y1 = bounds
    crop = image.crop((x0, y0, x1, y1))
    ax.imshow(crop, extent=[x0, x1, y1, y0], interpolation="bilinear")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_interpolated_domains(
    ax,
    df: pd.DataFrame,
    bounds: tuple[int, int, int, int],
    scale: int = 22,
    max_distance_px: float = 185,
) -> None:
    x0, x1, y0, y1 = bounds
    xs = np.arange(x0, x1 + scale, scale)
    ys = np.arange(y0, y1 + scale, scale)
    xx, yy = np.meshgrid(xs, ys)

    points = df[["pixel_x", "pixel_y"]].to_numpy(float)
    clusters = df["cluster"].to_numpy(int)
    tree = cKDTree(points)
    dist, idx = tree.query(np.c_[xx.ravel(), yy.ravel()], k=1)

    grid = clusters[idx].reshape(xx.shape).astype(float)
    grid[dist.reshape(xx.shape) > max_distance_px] = np.nan
    # Remove isolated single-spot islands while preserving the broader candidate domains.
    filled = np.where(np.isnan(grid), -1, grid).astype(int)
    smoothed = median_filter(filled, size=3)
    smoothed = smoothed.astype(float)
    smoothed[smoothed < 0] = np.nan

    cmap = ListedColormap([DOMAIN_COLORS[i] for i in sorted(DOMAIN_COLORS)])
    norm = BoundaryNorm(np.arange(-0.5, len(DOMAIN_COLORS) + 0.5, 1), cmap.N)
    ax.imshow(
        smoothed,
        origin="upper",
        extent=[x0, x1, y1, y0],
        cmap=cmap,
        norm=norm,
        alpha=0.28,
        interpolation="nearest",
    )
    for cluster, color in DOMAIN_COLORS.items():
        arr = (smoothed == cluster).astype(float)
        if np.nansum(arr) < 6:
            continue
        ax.contour(
            arr,
            levels=[0.5],
            colors=[color],
            linewidths=0.9,
            origin="upper",
            extent=[x0, x1, y1, y0],
        )


def save_figure() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(6.69, 3.85), dpi=600)
    titles = ["H1 manual annotation", "H2 candidate domains", "H3 candidate domains"]

    pred_tables = {slide: read_pred_clusters(slide) for slide in SLIDES}
    images = {slide: read_image(slide) for slide in SLIDES}

    for idx, slide in enumerate(SLIDES):
        ax = axes[idx]
        if slide == "H1":
            annotation = crop_nonwhite(read_annotation_image())
            ax.imshow(annotation)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            points = pred_tables[slide]
            bounds = panel_bounds(points, images[slide], margin=360)
            draw_background(ax, images[slide], bounds)
            draw_interpolated_domains(ax, pred_tables[slide], bounds)
        ax.set_title(titles[idx], fontsize=8.8, pad=3)
        ax.text(
            0.02,
            0.95,
            slide,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1.5},
        )

    label_handles = [
        Line2D([0], [0], color=color, lw=3, label=label.title())
        for label, color in LABEL_COLORS.items()
    ]
    domain_handles = [
        Line2D([0], [0], color=DOMAIN_COLORS[i], lw=3, label=f"Domain {i}")
        for i in sorted(DOMAIN_COLORS)
    ]

    leg1 = fig.legend(
        handles=label_handles,
        loc="lower left",
        bbox_to_anchor=(0.015, 0.01),
        ncol=2,
        frameon=False,
        title="H1 labels",
        title_fontsize=8.5,
        fontsize=8,
        handlelength=1.5,
        columnspacing=0.8,
    )
    fig.add_artist(leg1)
    fig.legend(
        handles=domain_handles,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.01),
        ncol=3,
        frameon=False,
        title="Candidate domains in H2/H3",
        title_fontsize=8.5,
        fontsize=8,
        handlelength=1.5,
        columnspacing=0.8,
    )

    fig.subplots_adjust(left=0.015, right=0.995, top=0.91, bottom=0.24, wspace=0.05)
    for ext in ("png", "pdf", "tif"):
        out = OUT_DIR / f"HER2ST_patient_H_annotation_candidate_domain_overlay.{ext}"
        fig.savefig(out, dpi=600)
    plt.close(fig)

    manifest = pd.DataFrame(
        [
            {
                "file": str(OUT_DIR / f"HER2ST_patient_H_annotation_candidate_domain_overlay.{ext}"),
                "description": (
                    "H1 label-derived tissue-region contours and H2/H3 predicted-expression "
                    "candidate-domain contours over H&E."
                ),
            }
            for ext in ("png", "pdf", "tif")
        ]
    )
    manifest.to_csv(OUT_DIR / "HER2ST_patient_H_annotation_candidate_domain_overlay_manifest.csv", index=False)


if __name__ == "__main__":
    save_figure()
