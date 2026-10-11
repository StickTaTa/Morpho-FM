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
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle
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
    parser = argparse.ArgumentParser(description="Try image-processing filters for HER2ST periodic dense-map artifacts.")
    parser.add_argument("--slide", default="H1", choices=["H1", "H2", "H3"])
    parser.add_argument("--gene", default="ACTB")
    parser.add_argument("--period", type=float, default=None, help="Dense-grid artifact period. Defaults to estimated spot spacing.")
    parser.add_argument("--notch-radius", type=float, default=3.0)
    parser.add_argument("--notch-strength", type=float, default=0.98)
    parser.add_argument("--background-sigma", type=float, default=8.0)
    parser.add_argument("--tv-weight", type=float, default=0.08)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_dense(slide: str, gene: str) -> np.ndarray:
    pred_dir = RESULT_ROOT / f"train_{slide}_to_{slide}" / slide
    genes = json.loads((pred_dir / "pred_superres_selected_genes.json").read_text(encoding="utf-8"))
    if gene not in genes:
        raise ValueError(f"{gene} not found in selected genes: {genes}")
    maps = np.load(pred_dir / "pred_superres_selected.npy", mmap_mode="r")
    return np.asarray(maps[..., genes.index(gene)], dtype=np.float32)


def estimate_period(slide: str) -> float:
    locs = pd.read_csv(ROOT / "data" / "superres_cache" / slide / "locs.tsv", sep=r"\s+", engine="python")
    xs = np.sort(np.unique(np.round(locs["x"].to_numpy(dtype=float) / 16.0).astype(int)))
    ys = np.sort(np.unique(np.round(locs["y"].to_numpy(dtype=float) / 16.0).astype(int)))
    diffs = []
    for arr in (xs, ys):
        d = np.diff(arr)
        d = d[d > 4]
        if d.size:
            diffs.extend(d.tolist())
    if not diffs:
        return 16.0
    return float(np.median(diffs))


def normalized_gaussian(values: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    v = np.nan_to_num(values, nan=0.0).astype(np.float32)
    w = valid.astype(np.float32)
    num = gaussian_filter(v * w, sigma=sigma, mode="nearest")
    den = gaussian_filter(w, sigma=sigma, mode="nearest")
    out = np.full_like(v, np.nan, dtype=np.float32)
    keep = den > 1e-4
    out[keep] = num[keep] / den[keep]
    return out


def make_notch_mask(shape: tuple[int, int], period: float, radius: float, strength: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cy, cx = h // 2, w // 2
    fy = h / period
    fx = w / period
    mask = np.ones((h, w), dtype=np.float32)

    def apply_notch(y0: float, x0: float) -> None:
        nonlocal mask
        dist2 = (yy - y0) ** 2 + (xx - x0) ** 2
        notch = 1.0 - strength * np.exp(-dist2 / (2 * radius * radius))
        mask *= notch.astype(np.float32)

    # Axis-aligned grid harmonics. These are the usual frequency peaks for rectangular spot-grid artifacts.
    for m in range(1, 8):
        for sign in (-1, 1):
            if abs(sign * fx) < w / 2 - 2:
                apply_notch(cy, cx + sign * m * fx)
            if abs(sign * fy) < h / 2 - 2:
                apply_notch(cy + sign * m * fy, cx)
        if m * fx >= w / 2 - 2 and m * fy >= h / 2 - 2:
            break
    return np.clip(mask, 0.0, 1.0)


def fourier_notch_filter(raw: np.ndarray, period: float, radius: float, strength: float, background_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(raw)
    background = normalized_gaussian(raw, valid, sigma=background_sigma)
    fill_value = float(np.nanmedian(raw[valid])) if valid.any() else 0.0
    filled_raw = np.where(valid, raw, fill_value).astype(np.float32)
    filled_bg = np.where(np.isfinite(background), background, fill_value).astype(np.float32)
    high = filled_raw - filled_bg
    high -= np.mean(high[valid]) if valid.any() else np.mean(high)

    fft = np.fft.fftshift(np.fft.fft2(high))
    notch = make_notch_mask(raw.shape, period=period, radius=radius, strength=strength)
    filtered_high = np.real(np.fft.ifft2(np.fft.ifftshift(fft * notch))).astype(np.float32)
    out = filled_bg + filtered_high
    out[~valid] = np.nan
    return out.astype(np.float32), notch


def normalize_01(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    valid = np.isfinite(x)
    vmin, vmax = np.percentile(x[valid], [1, 99]) if valid.any() else (0.0, 1.0)
    if vmax <= vmin:
        vmax = vmin + 1.0
    y = np.clip((x - vmin) / (vmax - vmin), 0, 1)
    y[~valid] = 0
    return y.astype(np.float32), float(vmin), float(vmax)


def denoise_methods(raw: np.ndarray, tv_weight: float) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(raw)
    x01, vmin, vmax = normalize_01(raw)
    tv = denoise_tv_chambolle(x01, weight=tv_weight, channel_axis=None).astype(np.float32)
    try:
        bilateral = denoise_bilateral(
            x01,
            sigma_color=0.08,
            sigma_spatial=5,
            channel_axis=None,
        ).astype(np.float32)
    except TypeError:
        bilateral = denoise_bilateral(x01, sigma_color=0.08, sigma_spatial=5, multichannel=False).astype(np.float32)
    tv = tv * (vmax - vmin) + vmin
    bilateral = bilateral * (vmax - vmin) + vmin
    tv[~valid] = np.nan
    bilateral[~valid] = np.nan
    return tv.astype(np.float32), bilateral.astype(np.float32)


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


def render(slide: str, gene: str, maps: list[tuple[str, np.ndarray]], notch: np.ndarray, out_dir: Path, dpi: int) -> None:
    he = np.array(Image.open(ROOT / "data" / "superres_cache" / slide / "he.jpg"))
    up = [(title, upsample(arr, he.shape[:2])) for title, arr in maps]
    vmin, vmax = vrange(*(arr for _, arr in up))
    fig, axes = plt.subplots(1, len(up), figsize=(12.8, 3.2))
    for ax, (title, arr) in zip(axes, up):
        ax.imshow(he)
        im = ax.imshow(arr, cmap="viridis", vmin=vmin, vmax=vmax, alpha=np.where(np.isfinite(arr), 0.92, 0))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.01)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels(["low", "high"])
    fig.suptitle(f"{slide} {gene}: periodic artifact filtering trials", fontsize=10, y=0.99)
    fig.subplots_adjust(left=0.01, right=0.98, top=0.86, bottom=0.02, wspace=0.04)
    fig.savefig(out_dir / f"{slide}_{gene}_periodic_filter_trial.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{slide}_{gene}_periodic_filter_trial.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(1, 1, figsize=(4, 4))
    ax2.imshow(notch, cmap="gray", vmin=0, vmax=1)
    ax2.set_title("Fourier notch mask", fontsize=9)
    ax2.axis("off")
    fig2.savefig(out_dir / f"{slide}_{gene}_fourier_notch_mask.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig2)


def main() -> None:
    args = parse_args()
    raw = load_dense(args.slide, args.gene)
    period = args.period if args.period is not None else estimate_period(args.slide)
    notch_map, notch_mask = fourier_notch_filter(
        raw,
        period=period,
        radius=args.notch_radius,
        strength=args.notch_strength,
        background_sigma=args.background_sigma,
    )
    tv_map, bilateral_map = denoise_methods(raw, tv_weight=args.tv_weight)

    paper = find_paper_dir()
    out_dir = paper / "Figure" / "Figure_6" / "HER2ST_patient_H" / "11_dense_periodic_artifact_filter_trial"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{args.slide}_{args.gene}_fourier_notch.npy", notch_map.astype(np.float32))
    np.save(out_dir / f"{args.slide}_{args.gene}_tv_denoise.npy", tv_map.astype(np.float32))
    np.save(out_dir / f"{args.slide}_{args.gene}_bilateral_denoise.npy", bilateral_map.astype(np.float32))
    np.save(out_dir / f"{args.slide}_{args.gene}_fourier_notch_mask.npy", notch_mask.astype(np.float32))

    maps = [
        ("Raw dense", raw),
        ("Fourier notch", notch_map),
        ("TV denoise", tv_map),
        ("Bilateral denoise", bilateral_map),
    ]
    render(args.slide, args.gene, maps, notch_mask, out_dir, args.dpi)
    stats = {
        "slide": args.slide,
        "gene": args.gene,
        "period_dense_grid": period,
        "notch_radius": args.notch_radius,
        "notch_strength": args.notch_strength,
        "background_sigma": args.background_sigma,
        "tv_weight": args.tv_weight,
        "note": "These are image-processing visualization trials for periodic artifact suppression. Quantitative analyses should use raw dense predictions.",
    }
    (out_dir / f"{args.slide}_{args.gene}_periodic_filter_trial_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(out_dir)


if __name__ == "__main__":
    main()
