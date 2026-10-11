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
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results" / "her2st_patient_h" / "dense_latent_clustering"
SLIDES = ("H1", "H2", "H3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine existing HER2ST dense latent clustering maps.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--slides", nargs="+", default=list(SLIDES), choices=SLIDES)
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def label_rgba(label_map: np.ndarray, out_hw: tuple[int, int], n_clusters: int) -> np.ndarray:
    encoded = np.where(np.isfinite(label_map), label_map.astype(np.int16), -1)
    up_img = Image.fromarray(encoded)
    up_img = up_img.resize((out_hw[1], out_hw[0]), resample=Image.Resampling.NEAREST)
    up = np.array(up_img)
    cmap = plt.get_cmap("tab10", n_clusters)
    rgba = np.zeros((*out_hw, 4), dtype=np.float32)
    valid = up >= 0
    rgba[valid] = cmap(up[valid].astype(int) % n_clusters)
    rgba[valid, 3] = 0.70
    return rgba


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(args.slides), figsize=(6.7, 2.35))
    if len(args.slides) == 1:
        axes = [axes]

    cluster_counts = []
    for ax, slide in zip(axes, args.slides):
        he = np.array(Image.open(root / "data" / "superres_cache" / slide / "he.jpg"))
        label_map = np.load(out_root / slide / f"{slide}_dense_latent_domain_map_display.npy")
        n_clusters = args.n_clusters
        if n_clusters is None:
            meta_path = out_root / slide / f"{slide}_dense_latent_cluster_meta.json"
            n_clusters = int(json.loads(meta_path.read_text(encoding="utf-8")).get("n_clusters", 7))
        cluster_counts.append(n_clusters)
        ax.imshow(he)
        ax.imshow(label_rgba(label_map, he.shape[:2], n_clusters))
        ax.set_title(f"{slide} (k={n_clusters})", fontsize=10)
        ax.axis("off")

    if len(set(cluster_counts)) == 1:
        n_clusters = cluster_counts[0]
        cmap = plt.get_cmap("tab10", n_clusters)
        handles = [mpatches.Patch(color=cmap(i), label=f"Domain {i}") for i in range(n_clusters)]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(n_clusters, 7),
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, -0.03),
        )
    fig.suptitle("HER2ST dense latent candidate domains", fontsize=11, y=0.98)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.18, wspace=0.03)
    for ext in ("pdf", "png", "tif"):
        fig.savefig(out_root / f"HER2ST_dense_latent_candidate_domains.{ext}", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {out_root}")


if __name__ == "__main__":
    main()
