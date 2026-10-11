from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill a selected HER2ST dense latent domain with nearest surrounding domains for display."
    )
    parser.add_argument(
        "--in-root",
        type=Path,
        default=ROOT / "results" / "her2st_patient_h" / "dense_latent_clustering_k10_raw",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "results" / "her2st_patient_h" / "dense_latent_clustering_k10_domain_fill",
    )
    parser.add_argument("--slide", default="H1")
    parser.add_argument("--target-domain", type=int, nargs="+", default=[4])
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def fill_domain(label_map: np.ndarray, target_domains: list[int]) -> np.ndarray:
    out = label_map.copy()
    finite = np.isfinite(out)
    encoded = np.full(out.shape, -1, dtype=np.int16)
    encoded[finite] = out[finite].astype(np.int16)
    target = finite & np.isin(encoded, np.asarray(target_domains, dtype=np.int16))
    source = finite & ~target
    if not target.any():
        raise ValueError(f"target domains {target_domains} are absent from label map")
    if not source.any():
        raise ValueError("no surrounding non-target domains are available for filling")

    # For each target pixel, copy the label of the nearest valid non-target pixel.
    _, indices = distance_transform_edt(~source, return_indices=True)
    nearest_y, nearest_x = indices
    out[target] = out[nearest_y[target], nearest_x[target]]
    return out


def upsample_rgba(label_map: np.ndarray, out_hw: tuple[int, int], n_clusters: int, alpha: float = 0.72) -> np.ndarray:
    encoded = np.full(label_map.shape, -1, dtype=np.int16)
    finite = np.isfinite(label_map)
    encoded[finite] = label_map[finite].astype(np.int16)
    up_img = Image.fromarray(encoded)
    up_img = up_img.resize((out_hw[1], out_hw[0]), resample=Image.Resampling.NEAREST)
    up = np.array(up_img)
    cmap = plt.get_cmap("tab10", n_clusters)
    rgba = np.zeros((*out_hw, 4), dtype=np.float32)
    valid = up >= 0
    rgba[valid] = cmap(up[valid].astype(int) % n_clusters)
    rgba[valid, 3] = alpha
    return rgba


def render_single(slide: str, he: np.ndarray, label_map: np.ndarray, n_clusters: int, title: str, out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6.7, 4.8))
    ax.imshow(he)
    ax.imshow(upsample_rgba(label_map, he.shape[:2], n_clusters))
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    cmap = plt.get_cmap("tab10", n_clusters)
    handles = [mpatches.Patch(color=cmap(i), label=f"Domain {i}") for i in range(n_clusters)]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_comparison(
    slide: str,
    he: np.ndarray,
    original: np.ndarray,
    filled: np.ndarray,
    target_label: str,
    n_clusters: int,
    out_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.75))
    for ax, label_map, title in zip(
        axes,
        [original, filled],
        [f"{slide} original k=10", f"{slide} Domain(s) {target_label} filled"],
    ):
        ax.imshow(he)
        ax.imshow(upsample_rgba(label_map, he.shape[:2], n_clusters))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    cmap = plt.get_cmap("tab10", n_clusters)
    handles = [mpatches.Patch(color=cmap(i), label=f"Domain {i}") for i in range(n_clusters)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.18, wspace=0.02)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.out_root / args.slide
    out_dir.mkdir(parents=True, exist_ok=True)

    in_dir = args.in_root / args.slide
    map_path = in_dir / f"{args.slide}_dense_latent_domain_map_display.npy"
    meta_path = in_dir / f"{args.slide}_dense_latent_cluster_meta.json"
    label_map = np.load(map_path)
    if meta_path.exists():
        n_clusters = int(json.loads(meta_path.read_text(encoding="utf-8")).get("n_clusters", 10))
    else:
        n_clusters = int(np.nanmax(label_map)) + 1

    target_tag = "_".join(str(x) for x in args.target_domain)
    filled = fill_domain(label_map, args.target_domain)
    np.save(out_dir / f"{args.slide}_dense_latent_domain_map_domain{target_tag}_filled.npy", filled)
    shutil.copy2(map_path, out_dir / f"{args.slide}_dense_latent_domain_map_original_display.npy")

    he = np.array(Image.open(ROOT / "data" / "superres_cache" / args.slide / "he.jpg"))
    render_single(
        args.slide,
        he,
        filled,
        n_clusters,
        f"{args.slide} dense latent domains, Domain(s) {target_tag} filled",
        out_dir / f"{args.slide}_dense_latent_domains_domain{target_tag}_filled",
        args.dpi,
    )
    render_comparison(
        args.slide,
        he,
        label_map,
        filled,
        target_tag,
        n_clusters,
        out_dir / f"{args.slide}_dense_latent_domains_domain{target_tag}_fill_comparison",
        args.dpi,
    )
    print(out_dir)


if __name__ == "__main__":
    main()
