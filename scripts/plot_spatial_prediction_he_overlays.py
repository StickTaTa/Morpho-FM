"""Plot representative true/predicted spot expression on H&E thumbnails."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = Path(r"C:\Users\Administrator\Desktop\人生的第4篇论文")
DEFAULT_REPO_OUT = ROOT / "results" / "paper_figures" / "rotating_single_benchmark_final"
DEFAULT_PAPER_OUT = PAPER_DIR / "Figure" / "benchmark_rebuild" / "rotating_single_benchmark_final"


PANELS = [
    {
        "cohort": "Prostate",
        "slide": "INT28",
        "train": "INT25",
        "gene": "KLK3",
        "result_dir": ROOT / "results" / "prostate_hest" / "morpho" / "single_superres" / "INT25_to_INT26_INT27_INT28" / "INT28",
    },
    {
        "cohort": "Kidney",
        "slide": "INT24",
        "train": "INT13",
        "gene": "B2M",
        "result_dir": ROOT / "results" / "kidney_hest" / "morpho" / "single_rotating" / "train_INT13_to_INT14_INT15_INT17_INT18_INT19_INT21_INT24" / "INT24",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-out", type=Path, default=DEFAULT_REPO_OUT)
    parser.add_argument("--paper-out", type=Path, default=DEFAULT_PAPER_OUT)
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
        }
    )


def decode(values: np.ndarray) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


def load_h5ad_spatial(slide: str) -> dict[str, object]:
    path = ROOT / "data" / "spatial_data" / f"{slide}.h5ad"
    with h5py.File(path, "r") as handle:
        obs = decode(handle["obs/_index"][:])
        image = handle["uns/spatial/ST/images/downscaled_fullres"][:]
        scale = float(handle["uns/spatial/ST/scalefactors/tissue_downscaled_fullres_scalef"][()])
        coords = handle["obsm/spatial"][:].astype(float) * scale
    return {"obs": obs, "image": image, "coords": coords}


def load_expression(panel: dict[str, object], gene_names: list[str]) -> dict[str, object]:
    gene = str(panel["gene"])
    gene_idx = gene_names.index(gene)
    result_dir = Path(panel["result_dir"])
    spot_ids = result_dir.joinpath("spot_ids.txt").read_text(encoding="utf-8").splitlines()
    pred = np.load(result_dir / "pred_bag.npy", mmap_mode="r")[:, gene_idx].astype(float)
    true = np.load(result_dir / "true_bag.npy", mmap_mode="r")[:, gene_idx].astype(float)
    spatial = load_h5ad_spatial(str(panel["slide"]))
    obs_to_idx = {spot: i for i, spot in enumerate(spatial["obs"])}
    keep = np.array([i for i, spot in enumerate(spot_ids) if spot in obs_to_idx], dtype=int)
    coord_idx = np.array([obs_to_idx[spot_ids[i]] for i in keep], dtype=int)
    return {
        "coords": spatial["coords"][coord_idx],
        "image": spatial["image"],
        "true": true[keep],
        "pred": pred[keep],
        "n": len(keep),
    }


def scale_values(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    denom = max(vmax - vmin, 1e-8)
    return np.clip((values - vmin) / denom, 0, 1)


def draw_overlay(ax: plt.Axes, image: np.ndarray, coords: np.ndarray, values: np.ndarray, title: str):
    ax.imshow(image)
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values,
        s=7,
        cmap="magma",
        vmin=0,
        vmax=1,
        alpha=0.78,
        linewidths=0,
    )
    ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold", pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return sc


def plot(outdir: Path) -> None:
    gene_names = (ROOT / "data" / "spatial_data" / "common_genes.txt").read_text(encoding="utf-8").splitlines()
    loaded = [load_expression(panel, gene_names) for panel in PANELS]
    fig = plt.figure(figsize=(184 / 25.4, 112 / 25.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.035], wspace=0.05, hspace=0.23)
    panel_letters = ["a", "b"]
    for row, (panel, data, letter) in enumerate(zip(PANELS, loaded, panel_letters)):
        values = np.concatenate([data["true"], data["pred"]])
        vmin, vmax = np.quantile(values, [0.02, 0.98])
        true_scaled = scale_values(data["true"], vmin, vmax)
        pred_scaled = scale_values(data["pred"], vmin, vmax)
        ax_true = fig.add_subplot(gs[row, 0])
        ax_pred = fig.add_subplot(gs[row, 1])
        cax = fig.add_subplot(gs[row, 2])
        prefix = f"{letter}  {panel['cohort']} {panel['slide']} ({panel['gene']})"
        sc = draw_overlay(ax_true, data["image"], data["coords"], true_scaled, f"{prefix} measured")
        draw_overlay(ax_pred, data["image"], data["coords"], pred_scaled, "predicted")
        cbar = fig.colorbar(sc, cax=cax)
        cbar.ax.tick_params(labelsize=6.3, width=0.45, length=2)
        cbar.set_label("Scaled expression", fontsize=6.8)
        ax_true.text(
            0.01,
            0.02,
            f"train {panel['train']} -> test {panel['slide']}; n={data['n']}",
            transform=ax_true.transAxes,
            fontsize=5.8,
            color="#F7F7F7",
            ha="left",
            va="bottom",
        )
    fig.subplots_adjust(left=0.015, right=0.975, top=0.94, bottom=0.035)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "supp_representative_spatial_he_overlay.pdf", bbox_inches="tight")
    fig.savefig(outdir / "supp_representative_spatial_he_overlay.png", dpi=420, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_style()
    for outdir in [args.repo_out, args.paper_out]:
        plot(outdir)
        print(f"saved {outdir}")


if __name__ == "__main__":
    main()
