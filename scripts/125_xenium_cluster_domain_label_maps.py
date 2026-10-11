from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(os.environ.get("MORPHO_VC_ROOT", r"E:\Morpho-FM")).resolve()
sys.path.append(str(ROOT / "src"))

from st_pipeline.data.gene_vocab import load_gene_vocab, map_genes_to_vocab  # noqa: E402
from st_pipeline.superres.feature_cache import load_cache_slide, normalize_slide  # noqa: E402


SECTION_CKPTS = {
    "XEN1": ROOT / "checkpoints" / "st_superres" / "xenium_cellfm",
    "XEN2": ROOT / "checkpoints" / "st_superres" / "xenium_cellfm_2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sections", nargs="+", default=["XEN1", "XEN2"], choices=["XEN1", "XEN2"])
    parser.add_argument("--n-clusters", type=int, default=15)
    parser.add_argument("--spot-point-size", type=float, default=2.0)
    parser.add_argument("--gray", default="#d9d9d9")
    parser.add_argument("--gray-alpha", type=float, default=0.18)
    parser.add_argument("--target-alpha", type=float, default=0.92)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def get_disk_mask(radius: float) -> np.ndarray:
    radius_ceil = int(np.ceil(radius))
    locs = np.meshgrid(
        np.arange(-radius_ceil, radius_ceil + 1),
        np.arange(-radius_ceil, radius_ceil + 1),
        indexing="ij",
    )
    locs = np.stack(locs, axis=-1)
    return (locs**2).sum(-1) <= radius**2


def load_section_locs(section: str) -> tuple[np.ndarray, tuple[int, int]]:
    ckpt_dir = SECTION_CKPTS[section]
    common_genes = json.loads((ckpt_dir / "common_genes.json").read_text(encoding="utf-8"))
    vocab = load_gene_vocab(ROOT / "assets" / "cellfm" / "expand_gene_info.csv")
    kept_genes, _ = map_genes_to_vocab(common_genes, vocab)

    embs, cnts, locs_px, radius_px, _, _ = load_cache_slide(section, root=ROOT, genes=kept_genes)
    embs_norm, _, _ = normalize_slide(embs, cnts)

    locs_px = np.asarray(locs_px, dtype=np.float32)
    locs_grid = np.round(locs_px / 16.0).astype(np.int64)
    mask = get_disk_mask(float(radius_px) / 16.0)
    h, w = embs_norm.shape[:2]
    shape = np.array(mask.shape)
    center = shape // 2
    r0 = -center
    r1 = shape - center
    valid = (
        (locs_grid[:, 0] + r0[0] >= 0)
        & (locs_grid[:, 0] + r1[0] <= h)
        & (locs_grid[:, 1] + r0[1] >= 0)
        & (locs_grid[:, 1] + r1[1] <= w)
    )
    finite_center = np.isfinite(embs_norm[locs_grid[:, 0], locs_grid[:, 1]]).all(-1)
    valid &= finite_center

    he_path = ROOT / "data" / "superres_cache" / section / "he.jpg"
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(he_path) as im:
        he_shape = (im.size[1], im.size[0])
    return locs_px[valid], he_shape


def domain_color(domain: int, n_clusters: int) -> tuple[float, float, float, float]:
    return plt.get_cmap("tab20", n_clusters)(domain)


def save_spot_domain_maps(
    section: str,
    locs_px: np.ndarray,
    labels: np.ndarray,
    he_shape: tuple[int, int],
    out_dir: Path,
    n_clusters: int,
    point_size: float,
    gray: str,
    gray_alpha: float,
    target_alpha: float,
    dpi: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = he_shape
    for domain in range(n_clusters):
        target = labels == domain
        fig, ax = plt.subplots(figsize=(8, 5.2))
        ax.scatter(
            locs_px[~target, 1],
            locs_px[~target, 0],
            s=point_size,
            color=gray,
            alpha=gray_alpha,
            edgecolors="none",
            rasterized=True,
        )
        ax.scatter(
            locs_px[target, 1],
            locs_px[target, 0],
            s=point_size,
            color=domain_color(domain, n_clusters),
            alpha=target_alpha,
            edgecolors="none",
            rasterized=True,
        )
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{section} Domain {domain} cell-position distribution", fontsize=10)
        ax.axis("off")
        fig.tight_layout(pad=0.05)
        fig.savefig(out_dir / f"Domain_{domain:02d}.pdf", bbox_inches="tight", dpi=dpi)
        fig.savefig(out_dir / f"Domain_{domain:02d}.png", bbox_inches="tight", dpi=dpi)
        plt.close(fig)


def make_full_slide_domain_rgb(
    cluster_map: np.ndarray,
    domain: int,
    n_clusters: int,
    gray_alpha: float,
    target_alpha: float,
) -> np.ndarray:
    rgb = np.ones((*cluster_map.shape, 3), dtype=np.float32)
    valid = np.isfinite(cluster_map)
    target = valid & (cluster_map.astype(np.float32) == float(domain))
    gray = np.array([0.85, 0.85, 0.85], dtype=np.float32)
    target_rgb = np.array(domain_color(domain, n_clusters)[:3], dtype=np.float32)

    rgb[valid] = (1.0 - gray_alpha) * rgb[valid] + gray_alpha * gray
    rgb[target] = (1.0 - target_alpha) * rgb[target] + target_alpha * target_rgb
    return rgb


def save_full_slide_domain_maps(
    section: str,
    cluster_map: np.ndarray,
    he_shape: tuple[int, int],
    out_dir: Path,
    n_clusters: int,
    gray_alpha: float,
    target_alpha: float,
    dpi: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = he_shape
    for domain in range(n_clusters):
        rgb = make_full_slide_domain_rgb(cluster_map, domain, n_clusters, gray_alpha, target_alpha)
        fig, ax = plt.subplots(figsize=(8, 5.2))
        ax.imshow(rgb, extent=(0, w, h, 0), interpolation="nearest")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{section} Domain {domain} full-slide distribution", fontsize=10)
        ax.axis("off")
        fig.tight_layout(pad=0.05)
        fig.savefig(out_dir / f"Domain_{domain:02d}.pdf", bbox_inches="tight", dpi=dpi)
        fig.savefig(out_dir / f"Domain_{domain:02d}.png", bbox_inches="tight", dpi=dpi)
        plt.close(fig)


def write_summary(section: str, labels: np.ndarray, cluster_map: np.ndarray, out_root: Path) -> None:
    rows = []
    valid_grid = np.isfinite(cluster_map)
    for domain in sorted(set(labels.tolist()) | set(np.unique(cluster_map[valid_grid]).astype(int).tolist())):
        rows.append(
            {
                "section": section,
                "domain": int(domain),
                "n_cell_position_points": int((labels == domain).sum()),
                "n_full_slide_grid_points": int((cluster_map[valid_grid].astype(int) == domain).sum()),
            }
        )
    import pandas as pd

    pd.DataFrame(rows).to_csv(out_root / "domain_label_map_summary.csv", index=False)


def process_section(args: argparse.Namespace, section: str) -> None:
    cluster_dir = ROOT / "results" / "st_superres" / "xenium_latent_clustering" / f"xenium_{section}"
    labels = np.load(cluster_dir / "latent_cluster_labels.npy")
    cluster_map = np.load(cluster_dir / "grid_latent_cluster_map.npy")
    locs_px, he_shape = load_section_locs(section)
    if len(locs_px) != len(labels):
        raise ValueError(f"{section}: loc count {len(locs_px):,} != label count {len(labels):,}")

    out_root = cluster_dir / "domain_label_maps"
    spot_dir = out_root / "cell_positions_no_he_gray_reference"
    full_dir = out_root / "full_slide_no_he_gray_reference"
    save_spot_domain_maps(
        section=section,
        locs_px=locs_px,
        labels=labels,
        he_shape=he_shape,
        out_dir=spot_dir,
        n_clusters=args.n_clusters,
        point_size=args.spot_point_size,
        gray=args.gray,
        gray_alpha=args.gray_alpha,
        target_alpha=args.target_alpha,
        dpi=args.dpi,
    )
    save_full_slide_domain_maps(
        section=section,
        cluster_map=cluster_map,
        he_shape=he_shape,
        out_dir=full_dir,
        n_clusters=args.n_clusters,
        gray_alpha=args.gray_alpha,
        target_alpha=args.target_alpha,
        dpi=args.dpi,
    )
    write_summary(section, labels, cluster_map, out_root)
    print(f"[done] {section}: {out_root}")


def main() -> None:
    args = parse_args()
    for section in args.sections:
        process_section(args, section)


if __name__ == "__main__":
    main()
