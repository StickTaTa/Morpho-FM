from __future__ import annotations

import argparse
import math
import os
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image


ROOT = Path(os.environ.get("MORPHO_VC_ROOT", r"E:\Morpho-FM")).resolve()
RESULT_DIR = ROOT / "results" / "st_superres" / "xenium_latent_clustering" / "domain_biology_enrichment"

SECTION_CONFIG = {
    "XEN1": {
        "label": "S1",
        "sheet": "Fig. 3e-j Xenium",
        "label_col": "Cluster",
        "target_domain": 13,
    },
    "XEN2": {
        "label": "S2",
        "sheet": "Sup Fig. 5 Xenium Rep 2",
        "label_col": "Celltype",
        "target_domain": 3,
    },
}

RELEVANT_LABELS = [
    "Myoepi_KRT15+",
]

ANNOTATION_COLORS = {
    "Myoepi_ACTA2+": "#9467bd",
    "Myoepi_KRT15+": "#e377c2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-label", default="Myoepi_KRT15+", choices=sorted(ANNOTATION_COLORS))
    parser.add_argument("--output-prefix", default=None)
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


def compute_valid_indices(section: str, locs_px: np.ndarray, n_labels: int) -> np.ndarray:
    if len(locs_px) == n_labels:
        return np.arange(len(locs_px), dtype=np.int64)

    cache_dir = ROOT / "data" / "superres_cache" / section
    locs_grid = np.round(locs_px / 16.0).astype(np.int64)
    with open(cache_dir / "embeddings-hist.pickle", "rb") as handle:
        emb_dict = pickle.load(handle)
    embs = np.concatenate([emb_dict["cls"], emb_dict["sub"], emb_dict["rgb"]]).transpose(1, 2, 0)

    radius_px = float((cache_dir / "radius.txt").read_text(encoding="utf-8").strip())
    mask = get_disk_mask(radius_px / 16.0)
    shape = np.array(mask.shape)
    center = shape // 2
    r0 = -center
    r1 = shape - center
    h, w = embs.shape[:2]
    valid = (
        (locs_grid[:, 0] + r0[0] >= 0)
        & (locs_grid[:, 0] + r1[0] <= h)
        & (locs_grid[:, 1] + r0[1] >= 0)
        & (locs_grid[:, 1] + r1[1] <= w)
    )
    valid &= np.isfinite(embs[locs_grid[:, 0], locs_grid[:, 1]]).all(-1)
    idx = np.where(valid)[0]
    if len(idx) != n_labels:
        raise ValueError(f"{section}: valid rows {len(idx):,} != latent labels {n_labels:,}")
    return idx


def load_section(section: str) -> dict[str, np.ndarray | tuple[float, float, float, float]]:
    cfg = SECTION_CONFIG[section]
    cache_dir = ROOT / "data" / "superres_cache" / section
    cluster_dir = ROOT / "results" / "st_superres" / "xenium_latent_clustering" / f"xenium_{section}"

    labels = np.load(cluster_dir / "latent_cluster_labels.npy").astype(int)
    locs = pd.read_csv(cache_dir / "locs.tsv", sep="\t")
    locs_px_all = np.stack(
        [locs["y"].to_numpy(dtype=np.float32), locs["x"].to_numpy(dtype=np.float32)],
        axis=-1,
    )
    valid_idx = compute_valid_indices(section, locs_px_all, len(labels))
    locs_px = locs_px_all[valid_idx]

    ann = pd.read_excel(ROOT / "notebooks" / "breast_cancer_info.xlsx", sheet_name=cfg["sheet"])
    if len(ann) != len(locs):
        raise ValueError(f"{section}: annotation rows {len(ann):,} != loc rows {len(locs):,}")
    annotation = ann[cfg["label_col"]].astype(str).to_numpy()[valid_idx]

    cnts = pd.read_csv(cache_dir / "cnts.tsv", sep="\t", usecols=["KRT14", "KRT5"])
    krt14 = cnts["KRT14"].to_numpy(dtype=np.float32)[valid_idx]
    krt5 = cnts["KRT5"].to_numpy(dtype=np.float32)[valid_idx]

    x = locs_px[:, 1]
    y = locs_px[:, 0]
    pad_x = (float(x.max()) - float(x.min())) * 0.025
    pad_y = (float(y.max()) - float(y.min())) * 0.025
    extent = (float(x.min() - pad_x), float(x.max() + pad_x), float(y.min() - pad_y), float(y.max() + pad_y))

    return {
        "coords": locs_px,
        "annotation": annotation,
        "labels": labels,
        "KRT14": krt14,
        "KRT5": krt5,
        "extent": extent,
    }


def robust_unit(values: np.ndarray) -> np.ndarray:
    log_values = np.log1p(np.asarray(values, dtype=np.float32))
    pos = log_values[np.isfinite(log_values) & (log_values > 0)]
    if pos.size == 0:
        return np.zeros_like(log_values, dtype=np.float32)
    vmin, vmax = np.percentile(pos, [1, 99])
    vmin = max(0.0, float(vmin))
    vmax = float(vmax)
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(pos))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return np.clip((log_values - vmin) / (vmax - vmin), 0.0, 1.0)


def format_axes(ax: plt.Axes, extent: tuple[float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_annotation(
    ax: plt.Axes,
    data: dict[str, np.ndarray],
    extent: tuple[float, float, float, float],
    annotation_label: str,
) -> None:
    coords = data["coords"]
    annotation = data["annotation"]
    ax.scatter(coords[:, 1], coords[:, 0], s=0.12, color="#d9d9d9", alpha=0.18, edgecolors="none", rasterized=True)
    mask = annotation == annotation_label
    if mask.any():
        ax.scatter(
            coords[mask, 1],
            coords[mask, 0],
            s=0.22,
            color=ANNOTATION_COLORS[annotation_label],
            alpha=0.9,
            edgecolors="none",
            rasterized=True,
        )
    format_axes(ax, extent)


def plot_expression(ax: plt.Axes, data: dict[str, np.ndarray], gene: str, extent: tuple[float, float, float, float]) -> None:
    coords = data["coords"]
    counts = np.asarray(data[gene], dtype=np.float32)
    norm = robust_unit(counts)
    positive = counts > 0
    ax.scatter(coords[:, 1], coords[:, 0], s=0.12, color="#d9d9d9", alpha=0.14, edgecolors="none", rasterized=True)
    ax.scatter(
        coords[positive, 1],
        coords[positive, 0],
        s=0.18,
        c=norm[positive],
        cmap="viridis",
        vmin=0,
        vmax=1,
        alpha=0.95,
        edgecolors="none",
        rasterized=True,
    )
    format_axes(ax, extent)


def plot_domain(ax: plt.Axes, data: dict[str, np.ndarray], domain: int, n_clusters: int, extent: tuple[float, float, float, float]) -> None:
    coords = data["coords"]
    labels = data["labels"]
    target = labels == domain
    cmap = plt.get_cmap("tab20", n_clusters)
    ax.scatter(coords[~target, 1], coords[~target, 0], s=0.12, color="#d9d9d9", alpha=0.18, edgecolors="none", rasterized=True)
    ax.scatter(
        coords[target, 1],
        coords[target, 0],
        s=0.26,
        color=cmap(domain),
        alpha=0.96,
        edgecolors="none",
        rasterized=True,
    )
    format_axes(ax, extent)


def find_paper_figure6_dir() -> Path:
    desktop = Path.home() / "Desktop"
    candidates = [p for p in desktop.iterdir() if p.is_dir() and (p / "Figure").exists()]
    for candidate in candidates:
        if (candidate / "main.tex").exists() or "论文" in candidate.name:
            fig6 = candidate / "Figure" / "Figure_6"
            fig6.mkdir(parents=True, exist_ok=True)
            return fig6
    raise FileNotFoundError("Could not find the manuscript Figure/Figure_6 directory on Desktop.")


def main() -> None:
    args = parse_args()
    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = (
            "KRT14_KRT5_myoepithelial_domain_evidence"
            if args.annotation_label == "Myoepi_KRT15+"
            else f"KRT14_KRT5_{args.annotation_label.replace('+', 'pos').replace('_', '')}_domain_evidence"
        )

    section_data = {section: load_section(section) for section in SECTION_CONFIG}
    n_clusters = 15

    width_in = 170 / 25.4
    height_in = 4.75
    fig, axes = plt.subplots(2, 4, figsize=(width_in, height_in), dpi=600)
    fig.patch.set_facecolor("white")
    col_titles = [
        "Annotation",
        r"$\it{KRT14}$",
        r"$\it{KRT5}$",
        "Enriched domain",
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9.5, pad=4)

    manifest_rows = []
    marker_summary = pd.read_csv(RESULT_DIR / "xenium_domain_KRT14_KRT5_marker_enrichment.csv")
    ranked = pd.read_csv(RESULT_DIR / "xenium_domain_dcis_myoepithelial_marker_ranked.csv")
    for row_idx, (section, cfg) in enumerate(SECTION_CONFIG.items()):
        data = section_data[section]
        extent = data["extent"]
        target_domain = cfg["target_domain"]
        plot_annotation(axes[row_idx, 0], data, extent, args.annotation_label)
        plot_expression(axes[row_idx, 1], data, "KRT14", extent)
        plot_expression(axes[row_idx, 2], data, "KRT5", extent)
        plot_domain(axes[row_idx, 3], data, target_domain, n_clusters, extent)
        axes[row_idx, 0].text(
            -0.08,
            0.5,
            cfg["label"],
            transform=axes[row_idx, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=10,
        )
        axes[row_idx, 3].text(
            0.02,
            0.04,
            f"Domain {target_domain}",
            transform=axes[row_idx, 3].transAxes,
            fontsize=7.2,
            va="bottom",
            ha="left",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
        sub = marker_summary[(marker_summary["section"] == section) & (marker_summary["domain"] == target_domain)]
        rank = ranked[(ranked["section"] == section) & (ranked["domain"] == target_domain)]
        manifest_rows.append(
            {
                "section": cfg["label"],
                "source_section": section,
                "target_domain": target_domain,
                "dominant_annotation_label": rank["dominant_annotation_label"].iloc[0] if len(rank) else "",
                "dcis_myoepithelial_fraction": float(rank["dcis_myoepithelial_fraction"].iloc[0]) if len(rank) else math.nan,
                "KRT14_positive_enrichment": float(sub[sub["gene"] == "KRT14"]["positive_fraction_enrichment"].iloc[0]),
                "KRT5_positive_enrichment": float(sub[sub["gene"] == "KRT5"]["positive_fraction_enrichment"].iloc[0]),
            }
        )

    plt.subplots_adjust(left=0.04, right=0.995, top=0.86, bottom=0.24, wspace=0.035, hspace=0.08)

    ann_handles = [mpatches.Patch(color=ANNOTATION_COLORS[args.annotation_label], label=args.annotation_label.replace("_", " "))]
    ann_handles.append(mpatches.Patch(color="#d9d9d9", label="Other cells"))
    fig.legend(
        handles=ann_handles,
        loc="lower left",
        bbox_to_anchor=(0.045, 0.055),
        ncol=4,
        frameon=False,
        fontsize=7.2,
        title="Annotation labels",
        title_fontsize=8,
        handlelength=0.9,
        columnspacing=0.9,
        labelspacing=0.3,
    )

    domain_handles = []
    cmap = plt.get_cmap("tab20", n_clusters)
    for section, cfg in SECTION_CONFIG.items():
        domain_handles.append(mpatches.Patch(color=cmap(cfg["target_domain"]), label=f"{cfg['label']} Domain {cfg['target_domain']}"))
    domain_handles.append(mpatches.Patch(color="#d9d9d9", label="Other domains"))
    fig.legend(
        handles=domain_handles,
        loc="lower center",
        bbox_to_anchor=(0.69, 0.055),
        ncol=3,
        frameon=False,
        fontsize=7.2,
        title="Highlighted latent domains",
        title_fontsize=8,
        handlelength=0.9,
        columnspacing=0.9,
        labelspacing=0.3,
    )

    cax = fig.add_axes([0.86, 0.09, 0.11, 0.015])
    sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap="viridis")
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["low", "high"])
    cbar.ax.tick_params(labelsize=7, length=0, pad=1)
    cbar.outline.set_linewidth(0.4)
    cax.set_title("Expression", fontsize=8, pad=2)

    out_result = RESULT_DIR / "KRT14_KRT5_myoepithelial_domain_evidence"
    out_result.mkdir(parents=True, exist_ok=True)
    fig6_dir = find_paper_figure6_dir() / "KRT14_KRT5_myoepithelial_DCiS" / "00_composite_evidence"
    fig6_dir.mkdir(parents=True, exist_ok=True)

    for out_dir in [out_result, fig6_dir]:
        for ext in ["png", "pdf", "tif"]:
            path = out_dir / f"{output_prefix}.{ext}"
            kwargs = {"facecolor": "white"}
            if ext in {"png", "tif"}:
                kwargs["dpi"] = 600
            if ext == "tif":
                kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
            fig.savefig(path, **kwargs)
        pd.DataFrame(manifest_rows).to_csv(out_dir / f"{output_prefix}_manifest.csv", index=False)
    plt.close(fig)

    print(pd.DataFrame(manifest_rows).to_string(index=False))
    print(f"[saved] {fig6_dir}")


if __name__ == "__main__":
    main()
