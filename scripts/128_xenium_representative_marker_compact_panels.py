from __future__ import annotations

import json
import os
import pickle
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


ROOT = Path(os.environ.get("MORPHO_VC_ROOT", r"E:\Morpho-FM")).resolve()
RESULT_ROOT = ROOT / "results" / "st_superres"
OUT_DIR = RESULT_ROOT / "xenium_extended" / "representative_marker_panels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAPER_DIR = Path(r"C:\Users\Administrator\Desktop\人生的第4篇论文\Figure\Figure_5\13_representative_6gene_panels")

GRID_FACTOR = 16
FIG_WIDTH_IN = 170 / 25.4
FIG_HEIGHT_IN = 9.2
CMAP = "viridis"

GENES = [
    ("FOXA1", "Tumour/luminal"),
    ("EPCAM", "Epithelial"),
    ("MKI67", "Proliferation"),
    ("LUM", "Stromal"),
    ("PECAM1", "Vascular"),
    ("PTPRC", "Immune"),
]

SECTION_RUNS = {
    "XEN1": {
        "label": "S1",
        "within": RESULT_ROOT / "xenium_cellfm_1" / "xenium_XEN1",
        "cross": RESULT_ROOT / "xenium_cellfm_2" / "xenium_XEN1",
        "dense_within": RESULT_ROOT / "xenium_extended" / "whole_slide" / "XEN1_within",
        "dense_cross": RESULT_ROOT / "xenium_extended" / "whole_slide" / "XEN2_to_XEN1",
    },
    "XEN2": {
        "label": "S2",
        "within": RESULT_ROOT / "xenium_cellfm_2" / "xenium_XEN2",
        "cross": RESULT_ROOT / "xenium_cellfm_1" / "xenium_XEN2",
        "dense_within": RESULT_ROOT / "xenium_extended" / "whole_slide" / "XEN2_within",
        "dense_cross": RESULT_ROOT / "xenium_extended" / "whole_slide" / "XEN1_to_XEN2",
    },
}


def get_disk_mask(radius: float) -> np.ndarray:
    radius_ceil = int(np.ceil(radius))
    locs = np.meshgrid(
        np.arange(-radius_ceil, radius_ceil + 1),
        np.arange(-radius_ceil, radius_ceil + 1),
        indexing="ij",
    )
    locs = np.stack(locs, axis=-1)
    return (locs**2).sum(-1) <= radius**2


def get_valid_coords(section: str) -> np.ndarray:
    cache_dir = ROOT / "data" / "superres_cache" / section
    locs_df = pd.read_csv(cache_dir / "locs.tsv", sep="\t")
    locs_px = np.stack([locs_df["y"].to_numpy(), locs_df["x"].to_numpy()], axis=-1).astype(np.float32)

    with open(cache_dir / "embeddings-hist.pickle", "rb") as handle:
        emb_dict = pickle.load(handle)
    embs = np.concatenate([emb_dict["cls"], emb_dict["sub"], emb_dict["rgb"]]).transpose(1, 2, 0)

    radius_px = float((cache_dir / "radius.txt").read_text(encoding="utf-8").strip())
    mask = get_disk_mask(radius_px / GRID_FACTOR)
    shape = np.array(mask.shape)
    center = shape // 2
    r0 = -center
    r1 = shape - center

    locs_grid = np.round(locs_px / float(GRID_FACTOR)).astype(np.int64)
    h, w = embs.shape[:2]
    valid = (
        (locs_grid[:, 0] + r0[0] >= 0)
        & (locs_grid[:, 0] + r1[0] <= h)
        & (locs_grid[:, 1] + r0[1] >= 0)
        & (locs_grid[:, 1] + r1[1] <= w)
    )
    valid &= np.isfinite(embs[locs_grid[:, 0], locs_grid[:, 1]]).all(-1)
    return locs_px[valid]


def robust_normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    if finite.size == 0:
        return out
    vmin, vmax = np.percentile(finite, [1, 99])
    vmin = max(0.0, float(vmin))
    vmax = float(vmax)
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(finite))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    out = (arr - vmin) / (vmax - vmin)
    return np.clip(out, 0.0, 1.0)


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return float("nan")
    a = a[mask] - np.mean(a[mask])
    b = b[mask] - np.mean(b[mask])
    denom = np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b))
    if denom == 0:
        return float("nan")
    return float(np.sum(a * b) / denom)


def scatter_values(ax, coords_xy: tuple[np.ndarray, np.ndarray], values: np.ndarray, extent: tuple[float, float, float, float]) -> None:
    x, y = coords_xy
    ax.scatter(
        x,
        y,
        c=robust_normalize(values),
        s=0.28,
        alpha=0.9,
        cmap=CMAP,
        vmin=0,
        vmax=1,
        edgecolors="none",
        rasterized=True,
    )
    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal")
    ax.axis("off")


def show_dense(ax, dense_map: np.ndarray, crop: tuple[int, int, int, int]) -> None:
    y0, y1, x0, x1 = crop
    norm_map = robust_normalize(dense_map[y0:y1, x0:x1])
    masked = np.ma.masked_invalid(norm_map)
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad(color="white", alpha=0.0)
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.axis("off")


def add_shared_colorbar(fig, axes) -> None:
    sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.018, pad=0.012)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["low", "high"])
    cbar.ax.tick_params(labelsize=7, length=0)
    cbar.outline.set_linewidth(0.5)


def measured_location_panel(section: str, fig_prefix: str) -> list[dict[str, object]]:
    cfg = SECTION_RUNS[section]
    section_label = cfg["label"]
    common_genes = json.loads((ROOT / "checkpoints" / "st_superres" / "xenium_cellfm" / "common_genes.json").read_text())
    gene_to_idx = {gene: idx for idx, gene in enumerate(common_genes)}

    coords = get_valid_coords(section)
    x = coords[:, 1]
    y = coords[:, 0]
    pad_x = (float(x.max()) - float(x.min())) * 0.03
    pad_y = (float(y.max()) - float(y.min())) * 0.03
    extent = (float(x.min() - pad_x), float(x.max() + pad_x), float(y.min() - pad_y), float(y.max() + pad_y))

    true_arr = np.load(cfg["within"] / "true_spot.npy", mmap_mode="r")
    within_arr = np.load(cfg["within"] / "pred_spot.npy", mmap_mode="r")
    cross_arr = np.load(cfg["cross"] / "pred_spot.npy", mmap_mode="r")

    fig, axes = plt.subplots(
        len(GENES),
        3,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        constrained_layout=False,
    )
    col_titles = ["Measured", "Within-section", "Reciprocal transfer"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=8, pad=3)

    rows = []
    for row_idx, (gene, category) in enumerate(GENES):
        gi = gene_to_idx[gene]
        true_values = np.asarray(true_arr[:, gi], dtype=np.float32)
        within_values = np.asarray(within_arr[:, gi], dtype=np.float32)
        cross_values = np.asarray(cross_arr[:, gi], dtype=np.float32)
        values = [true_values, within_values, cross_values]
        for col_idx, vals in enumerate(values):
            scatter_values(axes[row_idx, col_idx], (x, y), vals, extent)
        axes[row_idx, 0].text(
            -0.035,
            0.5,
            f"{gene}\n{category}",
            transform=axes[row_idx, 0].transAxes,
            ha="right",
            va="center",
            fontsize=7,
            linespacing=0.95,
        )
        rows.append(
            {
                "figure": fig_prefix,
                "section": section_label,
                "view": "observed_locations",
                "gene": gene,
                "category": category,
                "within_pearson": pearson_corr(true_values, within_values),
                "cross_pearson": pearson_corr(true_values, cross_values),
            }
        )

    fig.suptitle(f"{fig_prefix}: {section_label} observed-location marker maps", fontsize=9, y=0.995)
    fig.subplots_adjust(left=0.155, right=0.91, top=0.965, bottom=0.015, wspace=0.03, hspace=0.055)
    add_shared_colorbar(fig, axes)
    out_pdf = OUT_DIR / f"{fig_prefix}_{section_label}_observed_location_6genes.pdf"
    out_png = OUT_DIR / f"{fig_prefix}_{section_label}_observed_location_6genes.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=450, bbox_inches="tight")
    plt.close(fig)
    for r in rows:
        r["pdf"] = str(out_pdf)
        r["png"] = str(out_png)
    print(f"[saved] {out_pdf}")
    return rows


def dense_panel(section: str, fig_prefix: str) -> list[dict[str, object]]:
    cfg = SECTION_RUNS[section]
    section_label = cfg["label"]
    common_genes = json.loads((ROOT / "checkpoints" / "st_superres" / "xenium_cellfm" / "common_genes.json").read_text())
    gene_to_idx = {gene: idx for idx, gene in enumerate(common_genes)}
    panel_genes = json.loads((cfg["dense_within"] / "pred_superres_balanced_panel_genes.json").read_text())
    panel_gene_to_idx = {gene: idx for idx, gene in enumerate(panel_genes)}

    coords = get_valid_coords(section) / GRID_FACTOR
    x = coords[:, 1]
    y = coords[:, 0]
    true_arr = np.load(cfg["within"] / "true_spot.npy", mmap_mode="r")
    within_maps = np.load(cfg["dense_within"] / "pred_superres_balanced_panel.npy", mmap_mode="r")
    cross_maps = np.load(cfg["dense_cross"] / "pred_superres_balanced_panel.npy", mmap_mode="r")

    finite_mask = np.isfinite(within_maps[..., 0]) | np.isfinite(cross_maps[..., 0])
    yy, xx = np.where(finite_mask)
    pad = 8
    y0 = max(0, int(yy.min()) - pad)
    y1 = min(finite_mask.shape[0], int(yy.max()) + pad + 1)
    x0 = max(0, int(xx.min()) - pad)
    x1 = min(finite_mask.shape[1], int(xx.max()) + pad + 1)
    crop = (y0, y1, x0, x1)
    scatter_extent = (x0, x1, y0, y1)

    fig, axes = plt.subplots(
        len(GENES),
        3,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        constrained_layout=False,
    )
    col_titles = ["Measured locations", "Within-section dense", "Reciprocal dense"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=8, pad=3)

    rows = []
    for row_idx, (gene, category) in enumerate(GENES):
        gi = gene_to_idx[gene]
        pi = panel_gene_to_idx[gene]
        measured_values = np.asarray(true_arr[:, gi], dtype=np.float32)
        scatter_values(axes[row_idx, 0], (x, y), measured_values, scatter_extent)
        show_dense(axes[row_idx, 1], within_maps[..., pi], crop)
        show_dense(axes[row_idx, 2], cross_maps[..., pi], crop)
        axes[row_idx, 0].text(
            -0.035,
            0.5,
            f"{gene}\n{category}",
            transform=axes[row_idx, 0].transAxes,
            ha="right",
            va="center",
            fontsize=7,
            linespacing=0.95,
        )
        rows.append(
            {
                "figure": fig_prefix,
                "section": section_label,
                "view": "whole_section_dense",
                "gene": gene,
                "category": category,
            }
        )

    fig.suptitle(f"{fig_prefix}: {section_label} whole-section dense marker maps", fontsize=9, y=0.995)
    fig.subplots_adjust(left=0.155, right=0.91, top=0.965, bottom=0.015, wspace=0.03, hspace=0.055)
    add_shared_colorbar(fig, axes)
    out_pdf = OUT_DIR / f"{fig_prefix}_{section_label}_whole_section_dense_6genes.pdf"
    out_png = OUT_DIR / f"{fig_prefix}_{section_label}_whole_section_dense_6genes.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=450, bbox_inches="tight")
    plt.close(fig)
    for r in rows:
        r["pdf"] = str(out_pdf)
        r["png"] = str(out_png)
    print(f"[saved] {out_pdf}")
    return rows


def sync_to_paper() -> None:
    if PAPER_DIR.exists():
        shutil.rmtree(PAPER_DIR)
    shutil.copytree(OUT_DIR, PAPER_DIR)
    print(f"[copied] {OUT_DIR} -> {PAPER_DIR}")


def main() -> None:
    rows: list[dict[str, object]] = []
    rows.extend(measured_location_panel("XEN1", "Supplementary_Fig_61a"))
    rows.extend(measured_location_panel("XEN2", "Supplementary_Fig_61b"))
    rows.extend(dense_panel("XEN1", "Supplementary_Fig_62a"))
    rows.extend(dense_panel("XEN2", "Supplementary_Fig_62b"))
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT_DIR / "representative_6gene_panel_manifest.csv", index=False)
    sync_to_paper()


if __name__ == "__main__":
    main()
