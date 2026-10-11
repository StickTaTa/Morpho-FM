"""Plot Xenium cell-type annotation overlays for manuscript figures.

The published annotation workbook contains the section-1 annotation in
``Fig. 3e-j Xenium`` and the section-2 annotation in
``Sup Fig. 5 Xenium Rep 2``.  This script maps either sheet onto the
corresponding cached H&E image using the section-specific 10x alignment
matrix.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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


Image.MAX_IMAGE_PIXELS = None


SECTION_CONFIG = {
    "XEN1": {
        "section_folder": "section_1",
        "rep": 1,
        "sheet": "Fig. 3e-j Xenium",
        "label_col": "Cluster",
    },
    "XEN2": {
        "section_folder": "section_2",
        "rep": 2,
        "sheet": "Sup Fig. 5 Xenium Rep 2",
        "label_col": "Celltype",
    },
}

LABEL_ORDER = [
    "Stromal",
    "Invasive_Tumor",
    "DCIS 1",
    "DCIS 2",
    "Macrophages_1",
    "Endothelial",
    "Unlabeled",
    "CD4+_T_Cells",
    "Myoepi_ACTA2+",
    "CD8+_T_Cells",
    "B_Cells",
    "Prolif_Invasive_Tumor",
    "Myoepi_KRT15+",
    "Macrophages_2",
    "Perivascular-Like",
    "Stromal_&_T_Cell_Hybrid",
    "T_Cell_&_Tumor_Hybrid",
    "IRF7+_DCs",
    "LAMP3+_DCs",
    "Mast_Cells",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=sorted(SECTION_CONFIG), default="XEN2")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("MORPHO_VC_ROOT", r"E:\Morpho-FM")),
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=None,
        help="Annotation workbook. Defaults to notebooks/breast_cancer_info.xlsx.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.78)
    parser.add_argument(
        "--label-maps",
        action="store_true",
        help="Also write one H&E overlay per annotation label.",
    )
    parser.add_argument(
        "--label-maps-no-he",
        action="store_true",
        help=(
            "Also write one white-background coordinate map per annotation label, "
            "with non-target labels shown as a light gray reference."
        ),
    )
    return parser.parse_args()


def load_he_scale(cache_dir: Path) -> tuple[np.ndarray, float, float]:
    he_path = cache_dir / "he.jpg"
    meta_path = cache_dir / "cache_meta.json"
    he_scaled_path = cache_dir / "he-scaled.jpg"

    he_img = np.array(Image.open(he_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        cache_meta = json.load(f)
    with Image.open(he_scaled_path) as im_scaled:
        unpadded_w, unpadded_h = im_scaled.size

    scale_x = unpadded_w / cache_meta["fullres_width"]
    scale_y = unpadded_h / cache_meta["fullres_height"]
    return he_img, scale_x, scale_y


def map_annotation_to_he(
    df: pd.DataFrame,
    align_csv: Path,
    scale_x: float,
    scale_y: float,
    morph_pixel_size: float = 0.2125,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.loadtxt(align_csv, delimiter=",")
    matrix_inv = np.linalg.inv(matrix)

    x_morph = df["x_centroid"].to_numpy(dtype=float) / morph_pixel_size
    y_morph = df["y_centroid"].to_numpy(dtype=float) / morph_pixel_size
    coords = np.stack([x_morph, y_morph, np.ones_like(x_morph)], axis=0)
    xy = (matrix_inv @ coords)[:2].T
    return xy[:, 0] * scale_x, xy[:, 1] * scale_y


def get_label_palette(all_labels: list[str]) -> dict[str, tuple[float, float, float, float]]:
    ordered = [label for label in LABEL_ORDER if label in set(all_labels)]
    ordered += sorted(label for label in set(all_labels) if label not in LABEL_ORDER)
    tab20 = list(plt.get_cmap("tab20").colors)
    colors = {}
    for i, label in enumerate(ordered):
        rgb = tab20[i % len(tab20)]
        colors[label] = (*rgb, 1.0)
    return colors


def sanitize_filename(label: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(label)).strip("_")
    return name or "label"


def load_all_annotation_labels(excel: Path) -> list[str]:
    labels = []
    for cfg in SECTION_CONFIG.values():
        df = pd.read_excel(excel, sheet_name=cfg["sheet"], usecols=[cfg["label_col"]])
        labels.extend(df[cfg["label_col"]].astype(str).unique().tolist())
    return labels


def write_palette_csv(colors: dict[str, tuple[float, float, float, float]], out_dir: Path) -> None:
    rows = []
    for label in [x for x in LABEL_ORDER if x in colors] + sorted(set(colors) - set(LABEL_ORDER)):
        rgba = colors[label]
        rgb255 = tuple(int(round(v * 255)) for v in rgba[:3])
        rows.append(
            {
                "label": label,
                "r": rgb255[0],
                "g": rgb255[1],
                "b": rgb255[2],
                "hex": "#{:02x}{:02x}{:02x}".format(*rgb255),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "annotation_label_palette.csv", index=False)


def plot_single_label_maps(
    section: str,
    he_img: np.ndarray,
    df_plot: pd.DataFrame,
    x_plot: np.ndarray,
    y_plot: np.ndarray,
    label_col: str,
    labels: list[str],
    colors: dict[str, tuple[float, float, float, float]],
    out_dir: Path,
    dpi: int,
    point_size: float,
) -> None:
    label_dir = out_dir / "label_maps"
    label_dir.mkdir(parents=True, exist_ok=True)
    label_values = df_plot[label_col].astype(str).to_numpy()

    for label in labels:
        mask = label_values == label
        prefix = f"{sanitize_filename(label)}"
        pdf_path = label_dir / f"{prefix}.pdf"
        png_path = label_dir / f"{prefix}.png"
        fig, ax = plt.subplots(figsize=(10, 6.5))
        ax.imshow(he_img)
        ax.scatter(
            x_plot[mask],
            y_plot[mask],
            s=point_size,
            color=colors[label],
            alpha=0.88,
            edgecolors="none",
            rasterized=True,
        )
        ax.set_title(f"{section} {label} spatial distribution", fontsize=10)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(pdf_path, bbox_inches="tight", dpi=dpi)
        fig.savefig(png_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)


def plot_single_label_maps_no_he(
    section: str,
    he_shape: tuple[int, int],
    df_plot: pd.DataFrame,
    x_plot: np.ndarray,
    y_plot: np.ndarray,
    label_col: str,
    labels: list[str],
    colors: dict[str, tuple[float, float, float, float]],
    out_dir: Path,
    dpi: int,
    point_size: float,
) -> None:
    label_dir = out_dir / "label_maps_no_he_gray_reference"
    label_dir.mkdir(parents=True, exist_ok=True)
    label_values = df_plot[label_col].astype(str).to_numpy()
    h, w = he_shape

    for label in labels:
        mask = label_values == label
        prefix = f"{sanitize_filename(label)}"
        fig, ax = plt.subplots(figsize=(8, 5.2))
        ax.scatter(
            x_plot[~mask],
            y_plot[~mask],
            s=point_size,
            color="#d9d9d9",
            alpha=0.18,
            edgecolors="none",
            rasterized=True,
        )
        ax.scatter(
            x_plot[mask],
            y_plot[mask],
            s=point_size,
            color=colors[label],
            alpha=0.92,
            edgecolors="none",
            rasterized=True,
        )
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{section} {label} spatial distribution", fontsize=10)
        ax.axis("off")
        fig.tight_layout(pad=0.05)
        fig.savefig(label_dir / f"{prefix}.pdf", bbox_inches="tight", dpi=dpi)
        fig.savefig(label_dir / f"{prefix}.png", bbox_inches="tight", dpi=dpi)
        plt.close(fig)


def plot_overlay(section: str, root: Path, excel: Path, dpi: int, point_size: float, alpha: float) -> None:
    cfg = SECTION_CONFIG[section]
    data_dir = root / "data" / "Xenium_breast_cancer"
    cache_dir = root / "data" / "superres_cache" / section
    out_dir = root / "results" / "st_superres" / "xenium_annotation" / section
    out_dir.mkdir(parents=True, exist_ok=True)

    align_csv = (
        data_dir
        / cfg["section_folder"]
        / f"Xenium_FFPE_Human_Breast_Cancer_Rep{cfg['rep']}_he_imagealignment.csv"
    )
    he_img, scale_x, scale_y = load_he_scale(cache_dir)

    df = pd.read_excel(excel, sheet_name=cfg["sheet"])
    x_he, y_he = map_annotation_to_he(df, align_csv, scale_x, scale_y)

    inside = (
        (x_he >= 0)
        & (x_he < he_img.shape[1])
        & (y_he >= 0)
        & (y_he < he_img.shape[0])
    )
    df_plot = df.loc[inside].copy()
    x_plot = x_he[inside]
    y_plot = y_he[inside]

    label_col = cfg["label_col"]
    df_plot[label_col] = df_plot[label_col].astype(str)
    present = set(df_plot[label_col].unique())
    labels = [label for label in LABEL_ORDER if label in present]
    labels += sorted(present - set(labels))
    colors = get_label_palette(load_all_annotation_labels(excel))
    write_palette_csv(colors, out_dir)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(he_img)
    for label in labels:
        mask = df_plot[label_col].to_numpy() == label
        ax.scatter(
            x_plot[mask],
            y_plot[mask],
            s=point_size,
            color=colors[label],
            label=label,
            alpha=alpha,
            edgecolors="none",
            rasterized=True,
        )

    ax.legend(
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        markerscale=4,
        frameon=True,
        fontsize=8,
    )
    ax.set_title(f"Cell spatial annotation overlay - {section}", fontsize=10)
    ax.axis("off")
    fig.tight_layout()

    pdf_path = out_dir / "spatial_annotation_he_overlay.pdf"
    png_path = out_dir / "spatial_annotation_he_overlay.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=dpi)
    fig.savefig(png_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)

    summary = pd.DataFrame(
        [
            {
                "section": section,
                "sheet": cfg["sheet"],
                "label_column": cfg["label_col"],
                "n_annotated_cells": len(df),
                "n_cells_inside_he": int(inside.sum()),
                "fraction_inside_he": float(inside.mean()),
                "pdf": str(pdf_path),
                "png": str(png_path),
            }
        ]
    )
    summary.to_csv(out_dir / "spatial_annotation_he_overlay_summary.csv", index=False)
    if args_label_maps:
        plot_single_label_maps(
            section=section,
            he_img=he_img,
            df_plot=df_plot,
            x_plot=x_plot,
            y_plot=y_plot,
            label_col=label_col,
            labels=labels,
            colors=colors,
            out_dir=out_dir,
            dpi=dpi,
            point_size=point_size,
        )
    if args_label_maps_no_he:
        plot_single_label_maps_no_he(
            section=section,
            he_shape=he_img.shape[:2],
            df_plot=df_plot,
            x_plot=x_plot,
            y_plot=y_plot,
            label_col=label_col,
            labels=labels,
            colors=colors,
            out_dir=out_dir,
            dpi=dpi,
            point_size=point_size,
        )
    print(summary.to_string(index=False))


args_label_maps = False
args_label_maps_no_he = False


def main() -> None:
    global args_label_maps, args_label_maps_no_he
    args = parse_args()
    args_label_maps = args.label_maps
    args_label_maps_no_he = args.label_maps_no_he
    root = args.root.resolve()
    excel = args.excel or (root / "notebooks" / "breast_cancer_info.xlsx")
    plot_overlay(args.section, root, excel, args.dpi, args.point_size, args.alpha)


if __name__ == "__main__":
    main()
