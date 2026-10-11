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
from scipy.ndimage import generic_filter, uniform_filter
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
    parser = argparse.ArgumentParser(description="Compare local mean/median/trimmed-mean views for HER2ST dense artifact removal.")
    parser.add_argument("--slide", default="H1", choices=["H1", "H2", "H3"])
    parser.add_argument("--gene", default="ACTB")
    parser.add_argument("--window", type=int, default=11)
    parser.add_argument("--trim", type=float, default=0.15)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_dense(slide: str, gene: str) -> np.ndarray:
    pred_dir = RESULT_ROOT / f"train_{slide}_to_{slide}" / slide
    genes = json.loads((pred_dir / "pred_superres_selected_genes.json").read_text(encoding="utf-8"))
    if gene not in genes:
        raise ValueError(f"{gene} not found in selected genes: {genes}")
    maps = np.load(pred_dir / "pred_superres_selected.npy", mmap_mode="r")
    return np.asarray(maps[..., genes.index(gene)], dtype=np.float32)


def local_mean(values: np.ndarray, size: int) -> np.ndarray:
    finite = np.isfinite(values)
    val = np.nan_to_num(values, nan=0.0).astype(np.float32)
    w = finite.astype(np.float32)
    num = uniform_filter(val * w, size=size, mode="nearest")
    den = uniform_filter(w, size=size, mode="nearest")
    out = np.full_like(val, np.nan, dtype=np.float32)
    keep = den > 1e-6
    out[keep] = num[keep] / den[keep]
    return out


def local_median(values: np.ndarray, size: int) -> np.ndarray:
    finite = np.isfinite(values)
    fill = float(np.nanmedian(values[finite])) if finite.any() else 0.0
    x = np.where(finite, values, fill).astype(np.float32)
    out = generic_filter(x, np.median, size=size, mode="nearest")
    out[~finite] = np.nan
    return out.astype(np.float32)


def local_trimmed_mean(values: np.ndarray, size: int, trim: float) -> np.ndarray:
    finite = np.isfinite(values)
    fill = float(np.nanmedian(values[finite])) if finite.any() else 0.0
    x = np.where(finite, values, fill).astype(np.float32)

    def trimmed(v):
        v = np.sort(v)
        n = len(v)
        cut = int(round(n * trim))
        if cut > 0 and n - 2 * cut > 1:
            v = v[cut:-cut]
        return float(np.mean(v))

    out = generic_filter(x, trimmed, size=size, mode="nearest")
    out[~finite] = np.nan
    return out.astype(np.float32)


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


def main() -> None:
    args = parse_args()
    raw = load_dense(args.slide, args.gene)
    mean_map = local_mean(raw, size=args.window)
    median_map = local_median(raw, size=args.window)
    trimmed_map = local_trimmed_mean(raw, size=args.window, trim=args.trim)

    paper = find_paper_dir()
    out_dir = paper / "Figure" / "Figure_6" / "HER2ST_patient_H" / "10_dense_artifact_mean_median_trial"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{args.slide}_{args.gene}_local_mean_w{args.window}.npy", mean_map.astype(np.float32))
    np.save(out_dir / f"{args.slide}_{args.gene}_local_median_w{args.window}.npy", median_map.astype(np.float32))
    np.save(out_dir / f"{args.slide}_{args.gene}_local_trimmed_mean_w{args.window}_trim{args.trim}.npy", trimmed_map.astype(np.float32))

    he = np.array(Image.open(ROOT / "data" / "superres_cache" / args.slide / "he.jpg"))
    panels = [
        ("Raw dense", raw),
        (f"Local mean (w={args.window})", mean_map),
        (f"Local median (w={args.window})", median_map),
        (f"Trimmed mean (trim={args.trim:g})", trimmed_map),
    ]
    up = [(title, upsample(arr, he.shape[:2])) for title, arr in panels]
    vmin, vmax = vrange(*(arr for _, arr in up))

    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.2))
    for ax, (title, arr) in zip(axes, up):
        ax.imshow(he)
        im = ax.imshow(arr, cmap="viridis", vmin=vmin, vmax=vmax, alpha=np.where(np.isfinite(arr), 0.92, 0))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.01)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels(["low", "high"])
    fig.suptitle(f"{args.slide} {args.gene}: local background views for dense artifact suppression", fontsize=10, y=0.99)
    fig.subplots_adjust(left=0.01, right=0.98, top=0.86, bottom=0.02, wspace=0.04)
    fig.savefig(out_dir / f"{args.slide}_{args.gene}_mean_median_trimmed_trial.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{args.slide}_{args.gene}_mean_median_trimmed_trial.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    stats = {
        "slide": args.slide,
        "gene": args.gene,
        "window": args.window,
        "trim": args.trim,
        "note": "Local background views are visualization diagnostics only. Quantitative analyses should use raw dense predictions.",
    }
    (out_dir / f"{args.slide}_{args.gene}_mean_median_trimmed_trial_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
