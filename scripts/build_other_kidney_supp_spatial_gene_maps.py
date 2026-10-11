from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


Image.MAX_IMAGE_PIXELS = None

ROOT = Path(r"E:\Morpho-FM")
PAPER_ROOT = Path(r"C:\Users\Administrator\Desktop\人生的第4篇论文")

DEFAULT_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_REPO_OUT = ROOT / "results" / "other_kidney" / f"supp_spatial_gene_maps_{DEFAULT_TAG}"
DEFAULT_PAPER_OUT = PAPER_ROOT / "Figure" / f"supp_other_kidney_spatial_gene_maps_{DEFAULT_TAG}"

PANELS = [
    {"slide": "GSM6415706_GLMF1", "slide_label": "Slide 1", "gene": "SDHB"},
    {"slide": "GSM6415706_GLMF1", "slide_label": "Slide 1", "gene": "TFEB"},
    {"slide": "GSM6415706_GLMF2", "slide_label": "Slide 2", "gene": "SMARCB1"},
    {"slide": "GSM6415706_GLMF2", "slide_label": "Slide 2", "gene": "SDHB"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-out", type=Path, default=DEFAULT_REPO_OUT)
    parser.add_argument("--paper-out", type=Path, default=DEFAULT_PAPER_OUT)
    parser.add_argument("--no-paper-copy", action="store_true")
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.55,
            "axes.titlepad": 5,
            "savefig.transparent": False,
        }
    )


def load_slide(slide: str) -> dict[str, object]:
    result_dir = ROOT / "results" / "other_kidney" / slide / "multi"
    cache_dir = ROOT / "data" / "other_kidney_cache" / slide
    genes = (result_dir / "kept_genes.txt").read_text(encoding="utf-8").splitlines()
    locs = np.load(result_dir / "locs_px.npy").astype(np.float32, copy=False)
    pred = np.load(result_dir / "pred_bag.npy", mmap_mode="r")
    true = np.load(result_dir / "true_bag.npy", mmap_mode="r")
    he = Image.open(cache_dir / "he.jpg").convert("RGB")
    return {
        "genes": genes,
        "gene_to_idx": {gene: i for i, gene in enumerate(genes)},
        "locs": locs,
        "pred": pred,
        "true": true,
        "he": he,
    }


def bounded_interval(center: float, size: float, upper: int) -> tuple[int, int]:
    size = min(float(size), float(upper))
    start = int(round(center - size / 2.0))
    end = int(round(center + size / 2.0))
    if start < 0:
        end -= start
        start = 0
    if end > upper:
        start -= end - upper
        end = upper
    start = max(0, start)
    return start, end


def crop_for_locs(
    he: Image.Image,
    locs_yx: np.ndarray,
    max_width: int = 1250,
    target_aspect: float = 1.13,
) -> dict[str, object]:
    y = locs_yx[:, 0]
    x = locs_yx[:, 1]
    x_span = float(np.nanmax(x) - np.nanmin(x))
    y_span = float(np.nanmax(y) - np.nanmin(y))
    margin = max(420.0, 0.055 * max(x_span, y_span))
    left = max(0, int(np.floor(np.nanmin(x) - margin)))
    right = min(he.width, int(np.ceil(np.nanmax(x) + margin)))
    top = max(0, int(np.floor(np.nanmin(y) - margin)))
    bottom = min(he.height, int(np.ceil(np.nanmax(y) + margin)))
    width = float(right - left)
    height = float(bottom - top)
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    if height > 0 and width / height < target_aspect:
        left, right = bounded_interval(center_x, target_aspect * height, he.width)
    elif width > 0 and width / height > target_aspect:
        top, bottom = bounded_interval(center_y, width / target_aspect, he.height)
    crop = he.crop((left, top, right, bottom))
    scale = 1.0
    if crop.width > max_width:
        scale = max_width / float(crop.width)
        crop = crop.resize((max_width, int(round(crop.height * scale))), Image.Resampling.LANCZOS)
    coords = np.column_stack(((x - left) * scale, (y - top) * scale)).astype(np.float32)
    return {"image": np.asarray(crop), "coords": coords, "scale": scale, "bbox": (left, top, right, bottom)}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def robust_scale(true_vals: np.ndarray, pred_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true_log = np.log1p(np.clip(np.asarray(true_vals, dtype=np.float32), 0, None))
    pred_log = np.log1p(np.clip(np.asarray(pred_vals, dtype=np.float32), 0, None))
    ref = np.concatenate([true_log[np.isfinite(true_log)], pred_log[np.isfinite(pred_log)]])
    if ref.size == 0:
        return np.zeros_like(true_log), np.zeros_like(pred_log)
    vmin, vmax = np.nanpercentile(ref, [1, 99])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(ref))
        vmax = float(np.nanmax(ref))
    if vmax <= vmin:
        vmax = vmin + 1.0
    true_scaled = np.clip((true_log - vmin) / (vmax - vmin), 0, 1)
    pred_scaled = np.clip((pred_log - vmin) / (vmax - vmin), 0, 1)
    return true_scaled, pred_scaled


def draw_panel(
    ax: plt.Axes,
    image: np.ndarray,
    coords_xy: np.ndarray,
    values: np.ndarray,
    *,
    cmap: str = "viridis",
) -> None:
    ax.imshow(image, interpolation="nearest")
    ax.scatter(
        coords_xy[:, 0],
        coords_xy[:, 1],
        c=values,
        cmap=cmap,
        vmin=0,
        vmax=1,
        s=2.15,
        alpha=0.9,
        linewidths=0,
        rasterized=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_strip(ax: plt.Axes, label: str, *, rotation: int = 0, fontsize: float = 10.5) -> None:
    ax.set_facecolor("#d9d9d9")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, 0.5, label, ha="center", va="center", rotation=rotation, fontsize=fontsize, color="#4a4a4a")


def build_figure(out_dir: Path) -> Path:
    slides = {slide: load_slide(slide) for slide in sorted({p["slide"] for p in PANELS})}
    crops = {slide: crop_for_locs(data["he"], data["locs"]) for slide, data in slides.items()}

    panel_data: list[dict[str, object]] = []
    for panel in PANELS:
        slide = panel["slide"]
        gene = panel["gene"]
        data = slides[slide]
        gene_idx = data["gene_to_idx"][gene]
        true_vals = np.asarray(data["true"][:, gene_idx], dtype=np.float32)
        pred_vals = np.asarray(data["pred"][:, gene_idx], dtype=np.float32)
        true_scaled, pred_scaled = robust_scale(true_vals, pred_vals)
        panel_data.append(
            {
                **panel,
                "true_scaled": true_scaled,
                "pred_scaled": pred_scaled,
                "pearson": pearson(true_vals, pred_vals),
            }
        )

    fig = plt.figure(figsize=(11.4, 5.72), dpi=220)
    label_x = 0.017
    label_w = 0.034
    panel_left = 0.067
    panel_right = 0.935
    panel_gap = 0.016
    panel_w = (panel_right - panel_left - 3 * panel_gap) / 4
    row_h = 0.355
    top_y = 0.545
    bottom_y = 0.143
    strip_y = 0.035
    strip_h = 0.056
    cbar_x = 0.957
    cbar_w = 0.012

    add_strip(fig.add_axes([label_x, top_y, label_w, row_h]), "True", rotation=90, fontsize=10.8)
    add_strip(fig.add_axes([label_x, bottom_y, label_w, row_h]), "Prediction", rotation=90, fontsize=10.8)

    for col, item in enumerate(panel_data):
        crop = crops[item["slide"]]
        x0 = panel_left + col * (panel_w + panel_gap)
        ax_true = fig.add_axes([x0, top_y, panel_w, row_h])
        ax_pred = fig.add_axes([x0, bottom_y, panel_w, row_h])
        draw_panel(ax_true, crop["image"], crop["coords"], item["true_scaled"])
        draw_panel(ax_pred, crop["image"], crop["coords"], item["pred_scaled"])
        gene_math = rf"$\it{{{item['gene']}}}$"
        ax_true.set_title(f"{gene_math}\n(r={item['pearson']:.3f})", fontsize=10.6, color="#444444", linespacing=0.84)

    slide1_x = panel_left
    slide1_w = 2 * panel_w + panel_gap
    slide2_x = panel_left + 2 * (panel_w + panel_gap)
    slide2_w = 2 * panel_w + panel_gap
    add_strip(fig.add_axes([slide1_x, strip_y, slide1_w, strip_h]), "Slide 1", fontsize=11.3)
    add_strip(fig.add_axes([slide2_x, strip_y, slide2_w, strip_h]), "Slide 2", fontsize=11.3)

    cax = fig.add_axes([cbar_x, bottom_y, cbar_w, top_y + row_h - bottom_y])
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap="viridis")
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.ax.tick_params(labelsize=9.3, length=0, pad=2)
    cbar.outline.set_linewidth(0.75)

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "supp_other_kidney_remaining_spatial_gene_distribution"
    fig.savefig(base.with_suffix(".png"), dpi=450)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    plt.close(fig)

    pd.DataFrame(
        [
            {
                "slide": item["slide"],
                "slide_label": item["slide_label"],
                "gene": item["gene"],
                "pearson_raw": item["pearson"],
            }
            for item in panel_data
        ]
    ).to_csv(out_dir / "supp_other_kidney_remaining_spatial_gene_distribution_metrics.csv", index=False)
    return base.with_suffix(".png")


def main() -> None:
    args = parse_args()
    setup_style()
    repo_png = build_figure(args.repo_out)
    print(f"[SAVED] {repo_png}")
    print(f"[SAVED] {repo_png.with_suffix('.pdf')}")
    print(f"[SAVED] {repo_png.with_suffix('.svg')}")
    if not args.no_paper_copy:
        paper_png = build_figure(args.paper_out)
        print(f"[COPIED] {paper_png}")
        print(f"[COPIED] {paper_png.with_suffix('.pdf')}")
        print(f"[COPIED] {paper_png.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
