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
from skimage.transform import resize

ROOT = Path(__file__).resolve().parents[1]
FOLD = "loso_train_H2_H3_to_H1"
GENES = ["KRT19", "ERBB2", "KRT14", "KRT5", "MGP", "IGKC"]

Image.MAX_IMAGE_PIXELS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render compact H1 LOSO dense marker and dense cluster summaries.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--fold", default=FOLD)
    parser.add_argument("--section", default="H1")
    parser.add_argument("--genes", default=",".join(GENES))
    parser.add_argument(
        "--dense-root",
        type=Path,
        default=ROOT / "results" / "her2st_patient_h" / "whole_slide_loso",
    )
    parser.add_argument(
        "--cluster-root",
        type=Path,
        default=ROOT / "results" / "her2st_patient_h" / "dense_latent_clustering_loso",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "results" / "her2st_patient_h" / "leave_one_section_out_figures",
    )
    parser.add_argument("--dpi", type=int, default=900)
    return parser.parse_args()


def single_vrange(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(arr, [1, 99])
    vmin = max(0.0, float(vmin))
    vmax = float(vmax)
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def load_display_he(path: Path, max_dim: int = 4800) -> tuple[np.ndarray, float, float]:
    img = Image.open(path)
    orig_w, orig_h = img.size
    scale = min(1.0, float(max_dim) / float(max(orig_w, orig_h)))
    if scale < 1.0:
        img = img.resize((int(round(orig_w * scale)), int(round(orig_h * scale))), resample=Image.Resampling.LANCZOS)
    arr = np.array(img)
    return arr, arr.shape[1] / float(orig_w), arr.shape[0] / float(orig_h)


def upsample_map(map_small: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    valid = np.isfinite(map_small).astype(np.float32)
    values = np.nan_to_num(map_small, nan=0.0).astype(np.float32)
    up_values = resize(values, target_hw, order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)
    up_valid = resize(valid, target_hw, order=0, preserve_range=True, anti_aliasing=False) > 0.5
    up_values[~up_valid] = np.nan
    return up_values


def load_measured(root: Path, section: str, genes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    cnts_path = root / "data" / "superres_cache" / section / "cnts.tsv"
    locs_path = root / "data" / "superres_cache" / section / "locs.tsv"
    cnts = pd.read_csv(cnts_path, sep="\t", index_col=0)
    locs = pd.read_csv(locs_path, sep="\t", index_col=0)
    common = cnts.index.intersection(locs.index)
    cnts = cnts.loc[common, genes]
    locs = locs.loc[common]
    coords = locs[["x", "y"]].to_numpy(dtype=np.float32)
    return cnts.to_numpy(dtype=np.float32), coords


def plot_dense_markers(args: argparse.Namespace, genes: list[str], out_dir: Path) -> None:
    root = args.root.resolve()
    dense_dir = args.dense_root / args.fold / args.section
    pred_maps = np.load(dense_dir / "pred_superres_selected.npy")
    saved_genes = json.loads((dense_dir / "pred_superres_selected_genes.json").read_text(encoding="utf-8"))
    idx = [saved_genes.index(g) for g in genes]
    measured, coords = load_measured(root, args.section, genes)
    he, sx, sy = load_display_he(root / "data" / "superres_cache" / args.section / "he.jpg")
    coords = coords.copy()
    coords[:, 0] *= sx
    coords[:, 1] *= sy

    fig, axes = plt.subplots(len(genes), 2, figsize=(6.69, 8.25))
    for row, (gene, gi, mi) in enumerate(zip(genes, idx, range(len(genes)))):
        dense_up = upsample_map(np.asarray(pred_maps[..., gi], dtype=np.float32), he.shape[:2])
        mvmin, mvmax = single_vrange(measured[:, mi])
        dvmin, dvmax = single_vrange(dense_up)
        ax = axes[row, 0]
        ax.imshow(he)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=measured[:, mi],
            s=7,
            cmap="viridis",
            vmin=mvmin,
            vmax=mvmax,
            alpha=0.86,
            edgecolors="none",
            rasterized=True,
        )
        ax.set_ylabel(gene, fontsize=8, rotation=0, labelpad=22, va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Measured spots" if row == 0 else "", fontsize=9, pad=2)

        ax = axes[row, 1]
        ax.imshow(he)
        alpha = np.where(np.isfinite(dense_up), 0.93, 0.0)
        ax.imshow(dense_up, cmap="viridis", vmin=dvmin, vmax=dvmax, alpha=alpha)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Dense held-out prediction" if row == 0 else "", fontsize=9, pad=2)

    # One compact low/high scale for relative expression.
    cax = fig.add_axes([0.91, 0.16, 0.012, 0.68])
    grad = np.linspace(0, 1, 256)[:, None]
    cax.imshow(grad, cmap="viridis", origin="lower", aspect="auto")
    cax.set_xticks([])
    cax.set_yticks([0, 255])
    cax.set_yticklabels(["low", "high"], fontsize=7)
    cax.set_ylabel("Relative expression", fontsize=8)
    train_label = args.fold.removeprefix("loso_train_").split("_to_")[0].replace("_", "+")
    fig.suptitle(
        f"{args.section} held-out dense marker reconstruction (train {train_label})",
        fontsize=10,
        y=0.995,
    )
    fig.subplots_adjust(left=0.08, right=0.89, top=0.96, bottom=0.02, wspace=0.03, hspace=0.04)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.section}_loso_dense_marker_summary"
    fig.savefig(out_dir / f"{stem}.pdf", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.tif", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def label_rgba(label_map: np.ndarray, out_hw: tuple[int, int], n_clusters: int, alpha: float = 0.72) -> np.ndarray:
    encoded = np.full(label_map.shape, -1, dtype=np.int16)
    finite = np.isfinite(label_map)
    encoded[finite] = label_map[finite].astype(np.int16)
    img = Image.fromarray(encoded).resize((out_hw[1], out_hw[0]), resample=Image.Resampling.NEAREST)
    up = np.array(img)
    cmap = plt.get_cmap("tab10", n_clusters)
    rgba = np.zeros((*out_hw, 4), dtype=np.float32)
    valid = up >= 0
    rgba[valid] = cmap(up[valid].astype(int) % n_clusters)
    rgba[valid, 3] = alpha
    return rgba


def plot_dense_cluster(args: argparse.Namespace, out_dir: Path) -> None:
    root = args.root.resolve()
    cluster_dir = args.cluster_root / args.fold / args.section
    map_path = cluster_dir / f"{args.section}_dense_latent_domain_map_display.npy"
    meta = json.loads((cluster_dir / f"{args.section}_dense_latent_cluster_meta.json").read_text(encoding="utf-8"))
    label_map = np.load(map_path)
    n_clusters = int(meta.get("n_clusters", int(np.nanmax(label_map)) + 1))
    he, _, _ = load_display_he(root / "data" / "superres_cache" / args.section / "he.jpg")

    fig, ax = plt.subplots(figsize=(6.69, 4.15))
    ax.imshow(he)
    ax.imshow(label_rgba(label_map, he.shape[:2], n_clusters))
    train_label = args.fold.removeprefix("loso_train_").split("_to_")[0].replace("_", "+")
    ax.set_title(f"{args.section} dense latent candidate domains (train {train_label})", fontsize=10, pad=3)
    ax.axis("off")
    cmap = plt.get_cmap("tab10", n_clusters)
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", color=cmap(i), label=f"Domain {i}", markersize=7) for i in range(n_clusters)]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)
    stem = f"{args.section}_loso_dense_latent_cluster_summary"
    fig.savefig(out_dir / f"{stem}.pdf", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.tif", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    genes = [g.strip() for g in args.genes.split(",") if g.strip()]
    out_dir = args.out_root / args.section / args.fold
    plot_dense_markers(args, genes, out_dir)
    plot_dense_cluster(args, out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
