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
from scipy.ndimage import gaussian_filter
from skimage.transform import resize


ROOT = Path(r"E:\Morpho-FM")
RESULT_ROOT = ROOT / "results" / "her2st_patient_h" / "whole_slide"
FIG_ROOT = Path(r"C:\Users\Administrator\Desktop\人生的第4篇论文\Figure\Figure_6\HER2ST_patient_H")
OUT_ROOT = FIG_ROOT / "06_smoothed_dense_gene_maps"

SLIDES = ("H1", "H2", "H3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render HER2ST dense gene maps with NaN-aware Gaussian smoothing. "
            "Smoothing is for visualization only; quantitative analyses should use the original maps."
        )
    )
    parser.add_argument("--slides", nargs="+", default=list(SLIDES), choices=SLIDES)
    parser.add_argument(
        "--genes",
        default=None,
        help="Comma-separated subset of selected whole-slide genes to render. Defaults to all selected genes.",
    )
    parser.add_argument("--sigma", type=float, default=1.2, help="Gaussian sigma on the dense prediction grid.")
    parser.add_argument("--min-weight", type=float, default=0.08, help="Minimum smoothed valid-mask weight.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--format", nargs="+", default=["pdf", "png"], choices=["pdf", "png", "tif"])
    return parser.parse_args()


def vrange(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(arr, [1, 99])
    vmin = max(0.0, float(vmin))
    vmax = float(vmax)
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(arr))
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
    return vmin, vmax


def nan_gaussian_smooth(x: np.ndarray, sigma: float, min_weight: float) -> np.ndarray:
    valid = np.isfinite(x)
    values = np.nan_to_num(x, nan=0.0).astype(np.float32)
    weights = valid.astype(np.float32)
    smooth_values = gaussian_filter(values * weights, sigma=sigma, mode="nearest")
    smooth_weights = gaussian_filter(weights, sigma=sigma, mode="nearest")
    out = np.full_like(values, np.nan, dtype=np.float32)
    keep = smooth_weights > float(min_weight)
    out[keep] = smooth_values[keep] / smooth_weights[keep]
    return out


def upsample_map(map_small: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    valid = np.isfinite(map_small).astype(np.float32)
    values = np.nan_to_num(map_small, nan=0.0).astype(np.float32)
    up_values = resize(values, target_hw, order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)
    up_valid = resize(valid, target_hw, order=0, preserve_range=True, anti_aliasing=False) > 0.5
    up_values[~up_valid] = np.nan
    return up_values


def load_measured(slide: str, genes: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    cache_dir = ROOT / "data" / "superres_cache" / slide
    cnts = pd.read_csv(cache_dir / "cnts.tsv", sep="\t", index_col=0)
    locs = pd.read_csv(cache_dir / "locs.tsv", sep=r"\s+", engine="python").set_index("SpotID")
    common = cnts.index.intersection(locs.index)
    cnts = cnts.loc[common, genes]
    locs = locs.loc[common]
    locs_px = locs[["y", "x"]].to_numpy(dtype=np.float32)
    return locs_px, cnts


def render_gene(
    slide: str,
    gene: str,
    gene_idx: int,
    pred_maps: np.ndarray,
    he_img: np.ndarray,
    locs_px: np.ndarray,
    cnts: pd.DataFrame,
    out_dir: Path,
    sigma: float,
    min_weight: float,
    formats: list[str],
    dpi: int,
) -> dict[str, object]:
    dense_raw = np.asarray(pred_maps[..., gene_idx], dtype=np.float32)
    dense_smooth = nan_gaussian_smooth(dense_raw, sigma=sigma, min_weight=min_weight)
    dense_up = upsample_map(dense_smooth, he_img.shape[:2])

    measured = cnts[gene].to_numpy(dtype=np.float32)
    measured_vmin, measured_vmax = vrange(measured)
    dense_vmin, dense_vmax = vrange(dense_up)

    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.0))
    for ax in axes:
        ax.imshow(he_img)
        ax.axis("off")

    axes[0].set_title("H&E", fontsize=9)
    axes[1].set_title("Measured locations", fontsize=9)
    axes[2].set_title("Dense map (smoothed view)", fontsize=9)

    sc = axes[1].scatter(
        locs_px[:, 1],
        locs_px[:, 0],
        c=measured,
        s=3.0,
        alpha=0.82,
        cmap="viridis",
        vmin=measured_vmin,
        vmax=measured_vmax,
        edgecolors="none",
        rasterized=True,
    )
    cbar1 = fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.02)
    cbar1.ax.tick_params(labelsize=7)
    cbar1.set_ticks([measured_vmin, measured_vmax])
    cbar1.set_ticklabels(["low", "high"])

    alpha = np.where(np.isfinite(dense_up), 0.92, 0.0)
    im = axes[2].imshow(dense_up, cmap="viridis", vmin=dense_vmin, vmax=dense_vmax, alpha=alpha)
    cbar2 = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02)
    cbar2.ax.tick_params(labelsize=7)
    cbar2.set_ticks([dense_vmin, dense_vmax])
    cbar2.set_ticklabels(["low", "high"])

    fig.suptitle(f"{slide} | {gene}", fontsize=10, y=0.98)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02, wspace=0.05)

    paths: dict[str, str] = {}
    for fmt in formats:
        out_path = out_dir / f"{gene}_smoothed_dense_panel.{fmt}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        paths[fmt] = str(out_path)
    plt.close(fig)

    np.save(out_dir / f"{gene}_dense_smoothed.npy", dense_smooth.astype(np.float32))
    return {
        "slide": slide,
        "gene": gene,
        "sigma": sigma,
        "min_weight": min_weight,
        "measured_vmin": measured_vmin,
        "measured_vmax": measured_vmax,
        "smoothed_dense_vmin": dense_vmin,
        "smoothed_dense_vmax": dense_vmax,
        **{f"panel_{fmt}": path for fmt, path in paths.items()},
    }


def main() -> None:
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    for slide in args.slides:
        tag = f"train_{slide}_to_{slide}"
        pred_dir = RESULT_ROOT / tag / slide
        pred_path = pred_dir / "pred_superres_selected.npy"
        gene_path = pred_dir / "pred_superres_selected_genes.json"
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        genes = json.loads(gene_path.read_text(encoding="utf-8"))
        if args.genes:
            requested = [g.strip() for g in args.genes.split(",") if g.strip()]
            missing = [g for g in requested if g not in genes]
            if missing:
                raise ValueError(f"{slide} selected dense predictions do not contain genes: {missing}")
            gene_indices = [genes.index(g) for g in requested]
            genes_to_render = requested
        else:
            gene_indices = list(range(len(genes)))
            genes_to_render = genes
        pred_maps = np.load(pred_path, mmap_mode="r")
        he_img = np.array(Image.open(ROOT / "data" / "superres_cache" / slide / "he.jpg"))
        locs_px, cnts = load_measured(slide, genes_to_render)

        out_dir = OUT_ROOT / slide
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "visualization_note.txt").write_text(
            "NaN-aware Gaussian smoothing was applied only for figure visualization. "
            "All quantitative analyses should use the unsmoothed dense predictions.\n",
            encoding="utf-8",
        )
        print(f"[{slide}] rendering {len(genes_to_render)} genes -> {out_dir}")
        for gene_idx, gene in zip(gene_indices, genes_to_render):
            row = render_gene(
                slide=slide,
                gene=gene,
                gene_idx=gene_idx,
                pred_maps=pred_maps,
                he_img=he_img,
                locs_px=locs_px,
                cnts=cnts,
                out_dir=out_dir,
                sigma=args.sigma,
                min_weight=args.min_weight,
                formats=args.format,
                dpi=args.dpi,
            )
            all_rows.append(row)
            print(f"  [saved] {gene}")

    manifest = OUT_ROOT / "smoothed_dense_manifest.csv"
    pd.DataFrame(all_rows).to_csv(manifest, index=False)
    print(f"[done] {manifest}")


if __name__ == "__main__":
    main()
