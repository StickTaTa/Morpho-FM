from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt, uniform_filter
from skimage.morphology import remove_small_objects


ROOT = Path(r"E:\Morpho-FM")
CLUSTER_ROOT = ROOT / "results" / "her2st_patient_h" / "dense_latent_clustering"
OUT_RESULT_ROOT = ROOT / "results" / "her2st_patient_h" / "dense_latent_clustering_clean_display"
SLIDES = ("H1", "H2", "H3")


def find_paper_dir() -> Path:
    desktop = Path.home() / "Desktop"
    for path in desktop.iterdir():
        if path.is_dir() and (path / "main.tex").exists() and (path / "Figure").exists():
            return path
    raise FileNotFoundError("Could not find manuscript directory under Desktop.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create cleaner display maps for HER2ST dense latent clusters using categorical spatial regularization."
    )
    parser.add_argument("--slides", nargs="+", default=list(SLIDES), choices=SLIDES)
    parser.add_argument("--window", type=int, default=9, help="Majority-filter window on the dense grid.")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--min-component", type=int, default=80, help="Remove same-label islands smaller than this grid area.")
    parser.add_argument("--alpha", type=float, default=0.72)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_maps(slide: str) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    slide_dir = CLUSTER_ROOT / slide
    raw = np.load(slide_dir / f"{slide}_dense_latent_domain_map_raw.npy")
    display = np.load(slide_dir / f"{slide}_dense_latent_domain_map_display.npy")
    meta = json.loads((slide_dir / f"{slide}_dense_latent_cluster_meta.json").read_text(encoding="utf-8"))
    return raw, display, meta


def majority_filter(label_map: np.ndarray, n_clusters: int, window: int, iterations: int) -> np.ndarray:
    labels = np.where(np.isfinite(label_map), label_map.astype(np.int16), -1)
    valid = labels >= 0
    out = labels.copy()
    for _ in range(iterations):
        scores = np.stack(
            [uniform_filter((out == k).astype(np.float32), size=window, mode="nearest") for k in range(n_clusters)],
            axis=0,
        )
        new_labels = np.argmax(scores, axis=0).astype(np.int16)
        out[valid] = new_labels[valid]
    smoothed = out.astype(np.float32)
    smoothed[~valid] = np.nan
    return smoothed


def remove_islands(label_map: np.ndarray, n_clusters: int, min_component: int) -> np.ndarray:
    labels = np.where(np.isfinite(label_map), label_map.astype(np.int16), -1)
    valid = labels >= 0
    cleaned = labels.copy()
    removed = np.zeros(valid.shape, dtype=bool)
    for k in range(n_clusters):
        mask = labels == k
        if mask.any():
            keep = remove_small_objects(mask, min_size=min_component, connectivity=2)
            drop = mask & ~keep
            cleaned[drop] = -1
            removed |= drop
    holes = valid & (cleaned < 0)
    if holes.any() and np.any(cleaned >= 0):
        _, inds = distance_transform_edt(cleaned < 0, return_indices=True)
        filled = cleaned[tuple(inds)]
        cleaned[holes] = filled[holes]
    out = cleaned.astype(np.float32)
    out[~valid] = np.nan
    return out


def clean_map(raw: np.ndarray, n_clusters: int, window: int, iterations: int, min_component: int) -> np.ndarray:
    out = majority_filter(raw, n_clusters=n_clusters, window=window, iterations=iterations)
    out = remove_islands(out, n_clusters=n_clusters, min_component=min_component)
    out = majority_filter(out, n_clusters=n_clusters, window=max(3, window // 2 * 2 + 1), iterations=1)
    return out.astype(np.float32)


def label_rgba(label_map: np.ndarray, out_hw: tuple[int, int], n_clusters: int, alpha: float) -> np.ndarray:
    labels = np.where(np.isfinite(label_map), label_map.astype(np.int16), -1)
    up_img = Image.fromarray(labels)
    up_img = up_img.resize((out_hw[1], out_hw[0]), resample=Image.Resampling.NEAREST)
    up = np.array(up_img)
    cmap = plt.get_cmap("tab10", n_clusters)
    rgba = np.zeros((*out_hw, 4), dtype=np.float32)
    valid = up >= 0
    rgba[valid] = cmap(up[valid].astype(int) % n_clusters)
    rgba[valid, 3] = alpha
    return rgba


def render_slide(
    slide: str,
    raw_display: np.ndarray,
    clean_display: np.ndarray,
    n_clusters: int,
    out_dir: Path,
    alpha: float,
    dpi: int,
) -> dict[str, object]:
    he = np.array(Image.open(ROOT / "data" / "superres_cache" / slide / "he.jpg"))
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.25))
    panels = [("Original display", raw_display), ("Clean display", clean_display)]
    for ax, (title, label_map) in zip(axes, panels):
        ax.imshow(he)
        ax.imshow(label_rgba(label_map, he.shape[:2], n_clusters, alpha=alpha))
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    cmap = plt.get_cmap("tab10", n_clusters)
    handles = [mpatches.Patch(color=cmap(i), label=f"Domain {i}") for i in range(n_clusters)]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(n_clusters, 7),
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(f"{slide} dense latent domains", fontsize=9, y=0.98)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.83, bottom=0.22, wspace=0.03)
    out_pdf = out_dir / f"{slide}_dense_latent_domains_clean_display.pdf"
    out_png = out_dir / f"{slide}_dense_latent_domains_clean_display.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {
        "slide": slide,
        "n_clusters": n_clusters,
        "raw_display_finite": int(np.isfinite(raw_display).sum()),
        "clean_display_finite": int(np.isfinite(clean_display).sum()),
        "panel_pdf": str(out_pdf),
        "panel_png": str(out_png),
    }


def render_combined(
    clean_maps: dict[str, tuple[np.ndarray, int]],
    out_dir: Path,
    alpha: float,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, len(clean_maps), figsize=(6.7, 2.0))
    if len(clean_maps) == 1:
        axes = [axes]
    for ax, (slide, (label_map, n_clusters)) in zip(axes, clean_maps.items()):
        he = np.array(Image.open(ROOT / "data" / "superres_cache" / slide / "he.jpg"))
        ax.imshow(he)
        ax.imshow(label_rgba(label_map, he.shape[:2], n_clusters, alpha=alpha))
        ax.set_title(f"{slide} (k={n_clusters})", fontsize=8)
        ax.axis("off")
    fig.suptitle("HER2ST dense latent candidate domains, clean display", fontsize=9, y=0.98)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.83, bottom=0.02, wspace=0.03)
    fig.savefig(out_dir / "HER2ST_dense_latent_candidate_domains_clean_display.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "HER2ST_dense_latent_candidate_domains_clean_display.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_slide(slide: str, args: argparse.Namespace, paper_out: Path) -> tuple[dict[str, object], np.ndarray, int]:
    raw, display, meta = load_maps(slide)
    n_clusters = int(meta.get("n_clusters", np.nanmax(raw) + 1))
    clean = clean_map(
        raw,
        n_clusters=n_clusters,
        window=args.window,
        iterations=args.iterations,
        min_component=args.min_component,
    )
    result_dir = OUT_RESULT_ROOT / slide
    result_dir.mkdir(parents=True, exist_ok=True)
    out_dir = paper_out / slide
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(result_dir / f"{slide}_dense_latent_domain_map_clean_display.npy", clean)
    np.save(out_dir / f"{slide}_dense_latent_domain_map_clean_display.npy", clean)
    out_meta = {
        "slide": slide,
        "n_clusters": n_clusters,
        "source_raw_map": str(CLUSTER_ROOT / slide / f"{slide}_dense_latent_domain_map_raw.npy"),
        "window": args.window,
        "iterations": args.iterations,
        "min_component": args.min_component,
        "note": (
            "Clean display applies categorical majority filtering and small-island removal to the dense latent domain map. "
            "It is intended for visualization; raw dense latent cluster assignments are unchanged."
        ),
    }
    (result_dir / f"{slide}_clean_display_meta.json").write_text(json.dumps(out_meta, indent=2), encoding="utf-8")
    (out_dir / f"{slide}_clean_display_meta.json").write_text(json.dumps(out_meta, indent=2), encoding="utf-8")
    row = render_slide(slide, display, clean, n_clusters, out_dir, alpha=args.alpha, dpi=args.dpi)
    print(f"[{slide}] clean display saved")
    return row, clean, n_clusters


def main() -> None:
    args = parse_args()
    paper = find_paper_dir()
    paper_out = paper / "Figure" / "Figure_6" / "HER2ST_patient_H" / "13_clean_dense_latent_clustering"
    paper_out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    combined: dict[str, tuple[np.ndarray, int]] = {}
    for slide in args.slides:
        row, clean, n_clusters = run_slide(slide, args, paper_out)
        rows.append(row)
        combined[slide] = (clean, n_clusters)
    render_combined(combined, paper_out, alpha=args.alpha, dpi=args.dpi)
    pd.DataFrame(rows).to_csv(paper_out / "clean_dense_latent_clustering_manifest.csv", index=False)
    print(f"[done] {paper_out}")


if __name__ == "__main__":
    main()
