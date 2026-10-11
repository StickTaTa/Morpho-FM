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


ROOT = Path(__file__).resolve().parents[1]
GENES = ["ERBB2", "KRT19", "CD24", "MGP", "CD74", "IGKC"]
FOLDS = {
    "H1": ("loso_train_H2_H3_to_H1", "H2+H3"),
    "H2": ("loso_train_H1_H3_to_H2", "H1+H3"),
    "H3": ("loso_train_H1_H2_to_H3", "H1+H2"),
}
SECTION_FOLDERS = {
    "H1": "16_loso_H1_transfer_summary",
    "H2": "16_loso_H2_transfer_summary",
    "H3": "16_loso_H3_transfer_summary",
}
Image.MAX_IMAGE_PIXELS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render HER2ST LOSO spot-level marker maps.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "results" / "her2st_patient_h" / "leave_one_section_out_figures_spot_markers",
    )
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=Path.home()
        / "Desktop"
        / "人生的第4篇论文"
        / "Figure"
        / "Figure_6"
        / "HER2ST_patient_H",
    )
    parser.add_argument("--genes", default=",".join(GENES))
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
        img = img.resize(
            (int(round(orig_w * scale)), int(round(orig_h * scale))),
            resample=Image.Resampling.LANCZOS,
        )
    arr = np.array(img)
    return arr, arr.shape[1] / float(orig_w), arr.shape[0] / float(orig_h)


def load_loso(root: Path, section: str, fold: str, genes: list[str]):
    result_dir = root / "results" / "her2st_patient_h" / "morpho" / "leave_one_section_out" / fold / section
    ckpt_dir = root / "checkpoints" / "her2st_patient_h" / "st_superres" / "leave_one_section_out" / fold
    all_genes = json.loads((ckpt_dir / "common_genes.json").read_text(encoding="utf-8"))
    gene_idx = [all_genes.index(g) for g in genes]
    pred = np.load(result_dir / "pred_bag.npy")[:, gene_idx]
    true = np.load(result_dir / "true_bag.npy")[:, gene_idx]
    spot_ids = [line.strip() for line in (result_dir / "spot_ids.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    corr = np.load(result_dir / "gene_corr_all.npy")
    gene_corr = {g: float(corr[all_genes.index(g)]) for g in genes}
    return true, pred, spot_ids, gene_corr


def load_coords(root: Path, section: str, spot_ids: list[str]) -> np.ndarray:
    locs_path = root / "data" / "superres_cache" / section / "locs.tsv"
    import pandas as pd

    locs = pd.read_csv(locs_path, sep="\t", index_col=0)
    return locs.loc[spot_ids, ["x", "y"]].to_numpy(dtype=np.float32)


def render_section(
    root: Path,
    out_dir: Path,
    section: str,
    fold: str,
    train_label: str,
    genes: list[str],
    dpi: int,
) -> None:
    true, pred, spot_ids, gene_corr = load_loso(root, section, fold, genes)
    coords = load_coords(root, section, spot_ids)
    he, sx, sy = load_display_he(root / "data" / "superres_cache" / section / "he.jpg")
    coords = coords.copy()
    coords[:, 0] *= sx
    coords[:, 1] *= sy

    fig, axes = plt.subplots(len(genes), 2, figsize=(6.69, 8.25))
    for row, gene in enumerate(genes):
        vmin_t, vmax_t = single_vrange(true[:, row])
        vmin_p, vmax_p = single_vrange(pred[:, row])

        ax = axes[row, 0]
        ax.imshow(he)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=true[:, row],
            s=9,
            cmap="viridis",
            vmin=vmin_t,
            vmax=vmax_t,
            alpha=0.9,
            edgecolors="none",
            rasterized=True,
        )
        ax.set_ylabel(gene, fontsize=8, rotation=0, labelpad=22, va="center", fontstyle="italic")
        ax.set_title("Measured spots" if row == 0 else "", fontsize=9, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[row, 1]
        ax.imshow(he)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=pred[:, row],
            s=9,
            cmap="viridis",
            vmin=vmin_p,
            vmax=vmax_p,
            alpha=0.9,
            edgecolors="none",
            rasterized=True,
        )
        ax.text(
            0.02,
            0.96,
            f"r={gene_corr[gene]:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.0},
        )
        ax.set_title("Predicted spots" if row == 0 else "", fontsize=9, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])

    cax = fig.add_axes([0.91, 0.16, 0.012, 0.68])
    grad = np.linspace(0, 1, 256)[:, None]
    cax.imshow(grad, cmap="viridis", origin="lower", aspect="auto")
    cax.set_xticks([])
    cax.set_yticks([0, 255])
    cax.set_yticklabels(["low", "high"], fontsize=7)
    cax.set_ylabel("Relative expression", fontsize=8)
    fig.suptitle(f"{section} held-out spot-level marker prediction (train {train_label})", fontsize=10, y=0.995)
    fig.subplots_adjust(left=0.08, right=0.89, top=0.96, bottom=0.02, wspace=0.03, hspace=0.04)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{section}_loso_spot_high_pcc_marker_summary"
    fig.savefig(out_dir / f"{stem}.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.tif", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    genes = [g.strip() for g in args.genes.split(",") if g.strip()]
    for section, (fold, train_label) in FOLDS.items():
        local_out = args.out_root / section / fold
        render_section(root, local_out, section, fold, train_label, genes, args.dpi)
        paper_out = args.paper_root / SECTION_FOLDERS[section]
        paper_out.mkdir(parents=True, exist_ok=True)
        for ext in ["pdf", "png", "tif"]:
            src = local_out / f"{section}_loso_spot_high_pcc_marker_summary.{ext}"
            (paper_out / src.name).write_bytes(src.read_bytes())
    print(args.out_root)


if __name__ == "__main__":
    main()
