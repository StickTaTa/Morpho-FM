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
from scipy.ndimage import binary_dilation, gaussian_filter, generic_filter
from skimage.transform import resize


ROOT = Path(r"E:\Morpho-FM")
RESULT_ROOT = ROOT / "results" / "her2st_patient_h" / "whole_slide"


def find_paper_dir() -> Path:
    desktop = Path.home() / "Desktop"
    for path in desktop.iterdir():
        if path.is_dir() and (path / "main.tex").exists() and (path / "Figure").exists():
            return path
    raise FileNotFoundError("Could not find manuscript directory under Desktop.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Try spot-footprint artifact inpainting for HER2ST dense maps.")
    parser.add_argument("--slide", default="H1", choices=["H1", "H2", "H3"])
    parser.add_argument("--gene", default="ACTB")
    parser.add_argument("--spot-radius", type=int, default=2, help="Grid-cell radius around spot centers to consider artifact footprint.")
    parser.add_argument("--inpaint-sigma", type=float, default=4.0, help="Gaussian sigma on dense grid for normalized inpainting.")
    parser.add_argument("--high-quantile", type=float, default=0.65, help="Positive residual quantile inside spot footprints for high-only mask.")
    parser.add_argument("--residual-quantile", type=float, default=0.82, help="Positive high-frequency residual quantile for grid-agnostic artifact mask.")
    parser.add_argument("--median-size", type=int, default=9, help="Grid-cell window for nanmedian background trial.")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def disk_offsets(radius: int) -> list[tuple[int, int]]:
    rr = int(radius)
    out = []
    for dy in range(-rr, rr + 1):
        for dx in range(-rr, rr + 1):
            if dy * dy + dx * dx <= rr * rr:
                out.append((dy, dx))
    return out


def spot_footprint_mask(slide: str, shape: tuple[int, int], radius: int) -> np.ndarray:
    locs = pd.read_csv(ROOT / "data" / "superres_cache" / slide / "locs.tsv", sep=r"\s+", engine="python")
    yy = np.rint(locs["y"].to_numpy(dtype=float) / 16.0).astype(int)
    xx = np.rint(locs["x"].to_numpy(dtype=float) / 16.0).astype(int)
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    offsets = disk_offsets(radius)
    for cy, cx in zip(yy, xx):
        for dy, dx in offsets:
            y = cy + dy
            x = cx + dx
            if 0 <= y < h and 0 <= x < w:
                mask[y, x] = True
    return mask


def normalized_gaussian(values: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    v = np.nan_to_num(values, nan=0.0).astype(np.float32)
    w = valid.astype(np.float32)
    num = gaussian_filter(v * w, sigma=sigma, mode="nearest")
    den = gaussian_filter(w, sigma=sigma, mode="nearest")
    out = np.full_like(v, np.nan, dtype=np.float32)
    keep = den > 1e-4
    out[keep] = num[keep] / den[keep]
    return out


def nanmedian_filter(values: np.ndarray, size: int) -> np.ndarray:
    finite = np.isfinite(values)
    fill = np.nanmedian(values[finite]) if finite.any() else 0.0
    x = np.where(finite, values, fill).astype(np.float32)

    def med(v):
        return float(np.median(v))

    out = generic_filter(x, med, size=size, mode="nearest")
    out[~finite] = np.nan
    return out.astype(np.float32)


def inpaint_map(raw: np.ndarray, replace_mask: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(raw)
    source = finite & (~replace_mask)
    background = normalized_gaussian(raw, source, sigma=sigma)
    out = raw.copy()
    fill = replace_mask & finite & np.isfinite(background)
    out[fill] = background[fill]
    return out, background


def upsample(map_small: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    valid = np.isfinite(map_small).astype(np.float32)
    values = np.nan_to_num(map_small, nan=0.0).astype(np.float32)
    up_values = resize(values, target_hw, order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)
    up_valid = resize(valid, target_hw, order=0, preserve_range=True, anti_aliasing=False) > 0.5
    up_values[~up_valid] = np.nan
    return up_values


def vrange(*arrays: np.ndarray) -> tuple[float, float]:
    arr = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(arr, [1, 99])
    vmin = max(0.0, float(vmin))
    vmax = float(vmax)
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def render_trial(slide: str, gene: str, raw: np.ndarray, high_mask: np.ndarray, all_mask: np.ndarray, high_corr: np.ndarray, all_corr: np.ndarray, out_dir: Path, dpi: int) -> None:
    he = np.array(Image.open(ROOT / "data" / "superres_cache" / slide / "he.jpg"))
    maps = {
        "Raw dense": raw,
        "High spot-footprints inpainted": high_corr,
        "All spot-footprints inpainted": all_corr,
    }
    up = {k: upsample(v, he.shape[:2]) for k, v in maps.items()}
    vmin, vmax = vrange(*up.values())
    mask_up = upsample(np.where(high_mask, 1.0, np.nan), he.shape[:2])

    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.2))
    for ax in axes:
        ax.imshow(he)
        ax.axis("off")

    axes[0].set_title("Raw dense", fontsize=9)
    axes[1].set_title("Detected high spot mask", fontsize=9)
    axes[2].set_title("High-mask inpaint", fontsize=9)
    axes[3].set_title("All-spot inpaint", fontsize=9)

    alpha0 = np.where(np.isfinite(up["Raw dense"]), 0.92, 0)
    im0 = axes[0].imshow(up["Raw dense"], cmap="viridis", vmin=vmin, vmax=vmax, alpha=alpha0)
    axes[1].imshow(np.ma.masked_invalid(mask_up), cmap="Reds", vmin=0, vmax=1, alpha=np.where(np.isfinite(mask_up), 0.75, 0))
    axes[2].imshow(up["High spot-footprints inpainted"], cmap="viridis", vmin=vmin, vmax=vmax, alpha=np.where(np.isfinite(up["High spot-footprints inpainted"]), 0.92, 0))
    axes[3].imshow(up["All spot-footprints inpainted"], cmap="viridis", vmin=vmin, vmax=vmax, alpha=np.where(np.isfinite(up["All spot-footprints inpainted"]), 0.92, 0))
    cbar = fig.colorbar(im0, ax=[axes[0], axes[2], axes[3]], fraction=0.018, pad=0.01)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels(["low", "high"])

    fig.suptitle(f"{slide} {gene} spot-footprint artifact inpainting trial", fontsize=10, y=0.99)
    fig.subplots_adjust(left=0.01, right=0.98, top=0.86, bottom=0.02, wspace=0.04)
    fig.savefig(out_dir / f"{slide}_{gene}_artifact_inpaint_trial.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{slide}_{gene}_artifact_inpaint_trial.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_residual_trial(
    slide: str,
    gene: str,
    raw: np.ndarray,
    residual_mask: np.ndarray,
    residual_corr: np.ndarray,
    median_corr: np.ndarray,
    out_dir: Path,
    dpi: int,
) -> None:
    he = np.array(Image.open(ROOT / "data" / "superres_cache" / slide / "he.jpg"))
    up_raw = upsample(raw, he.shape[:2])
    up_res = upsample(residual_corr, he.shape[:2])
    up_med = upsample(median_corr, he.shape[:2])
    up_mask = upsample(np.where(residual_mask, 1.0, np.nan), he.shape[:2])
    vmin, vmax = vrange(up_raw, up_res, up_med)

    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.2))
    for ax in axes:
        ax.imshow(he)
        ax.axis("off")
    axes[0].set_title("Raw dense", fontsize=9)
    axes[1].set_title("Detected high-frequency mask", fontsize=9)
    axes[2].set_title("Residual inpaint", fontsize=9)
    axes[3].set_title("Median background view", fontsize=9)

    im = axes[0].imshow(up_raw, cmap="viridis", vmin=vmin, vmax=vmax, alpha=np.where(np.isfinite(up_raw), 0.92, 0))
    axes[1].imshow(np.ma.masked_invalid(up_mask), cmap="Reds", vmin=0, vmax=1, alpha=np.where(np.isfinite(up_mask), 0.72, 0))
    axes[2].imshow(up_res, cmap="viridis", vmin=vmin, vmax=vmax, alpha=np.where(np.isfinite(up_res), 0.92, 0))
    axes[3].imshow(up_med, cmap="viridis", vmin=vmin, vmax=vmax, alpha=np.where(np.isfinite(up_med), 0.92, 0))
    cbar = fig.colorbar(im, ax=[axes[0], axes[2], axes[3]], fraction=0.018, pad=0.01)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels(["low", "high"])
    fig.suptitle(f"{slide} {gene} grid-agnostic high-frequency artifact trial", fontsize=10, y=0.99)
    fig.subplots_adjust(left=0.01, right=0.98, top=0.86, bottom=0.02, wspace=0.04)
    fig.savefig(out_dir / f"{slide}_{gene}_residual_artifact_trial.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{slide}_{gene}_residual_artifact_trial.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    tag = f"train_{args.slide}_to_{args.slide}"
    pred_dir = RESULT_ROOT / tag / args.slide
    genes = json.loads((pred_dir / "pred_superres_selected_genes.json").read_text(encoding="utf-8"))
    if args.gene not in genes:
        raise ValueError(f"{args.gene} not in selected genes: {genes}")
    gene_idx = genes.index(args.gene)
    pred_maps = np.load(pred_dir / "pred_superres_selected.npy", mmap_mode="r")
    raw = np.asarray(pred_maps[..., gene_idx], dtype=np.float32)

    spot_mask = spot_footprint_mask(args.slide, raw.shape, radius=args.spot_radius) & np.isfinite(raw)
    all_corr, bg = inpaint_map(raw, spot_mask, sigma=args.inpaint_sigma)
    residual = raw - bg
    pos = residual[spot_mask & np.isfinite(residual) & (residual > 0)]
    if pos.size == 0:
        threshold = np.inf
    else:
        threshold = float(np.quantile(pos, args.high_quantile))
    high_mask = spot_mask & np.isfinite(residual) & (residual >= threshold)
    # Slightly dilate detected high-center cells so replacement removes the visible footprint, not only the peak.
    high_mask = binary_dilation(high_mask, iterations=1) & spot_mask
    high_corr, _ = inpaint_map(raw, high_mask, sigma=args.inpaint_sigma)

    finite = np.isfinite(raw)
    low_background = normalized_gaussian(raw, finite, sigma=args.inpaint_sigma)
    residual = raw - low_background
    pos_res = residual[finite & np.isfinite(residual) & (residual > 0)]
    residual_threshold = float(np.quantile(pos_res, args.residual_quantile)) if pos_res.size else np.inf
    residual_mask = finite & np.isfinite(residual) & (residual >= residual_threshold)
    residual_mask = binary_dilation(residual_mask, iterations=1) & finite
    residual_corr = raw.copy()
    residual_corr[residual_mask] = low_background[residual_mask]
    median_corr = nanmedian_filter(raw, size=args.median_size)

    paper_dir = find_paper_dir()
    out_dir = paper_dir / "Figure" / "Figure_6" / "HER2ST_patient_H" / "09_dense_artifact_inpainting_trial"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{args.slide}_{args.gene}_raw_dense.npy", raw.astype(np.float32))
    np.save(out_dir / f"{args.slide}_{args.gene}_high_mask_inpainted.npy", high_corr.astype(np.float32))
    np.save(out_dir / f"{args.slide}_{args.gene}_all_spot_inpainted.npy", all_corr.astype(np.float32))
    np.save(out_dir / f"{args.slide}_{args.gene}_high_artifact_mask.npy", high_mask.astype(np.uint8))
    np.save(out_dir / f"{args.slide}_{args.gene}_residual_inpainted.npy", residual_corr.astype(np.float32))
    np.save(out_dir / f"{args.slide}_{args.gene}_residual_artifact_mask.npy", residual_mask.astype(np.uint8))
    np.save(out_dir / f"{args.slide}_{args.gene}_median_background.npy", median_corr.astype(np.float32))
    render_trial(args.slide, args.gene, raw, high_mask, spot_mask, high_corr, all_corr, out_dir, args.dpi)
    render_residual_trial(args.slide, args.gene, raw, residual_mask, residual_corr, median_corr, out_dir, args.dpi)
    stats = {
        "slide": args.slide,
        "gene": args.gene,
        "spot_radius_grid": args.spot_radius,
        "inpaint_sigma_grid": args.inpaint_sigma,
        "high_quantile": args.high_quantile,
        "spot_mask_pixels": int(spot_mask.sum()),
        "high_mask_pixels": int(high_mask.sum()),
        "high_residual_threshold": threshold,
        "residual_quantile": args.residual_quantile,
        "residual_artifact_pixels": int(residual_mask.sum()),
        "residual_artifact_threshold": residual_threshold,
        "median_size_grid": args.median_size,
        "note": "Artifact inpainting is a visualization diagnostic. Quantitative analyses should use raw dense predictions.",
    }
    (out_dir / f"{args.slide}_{args.gene}_artifact_inpaint_trial_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(out_dir)


if __name__ == "__main__":
    main()
