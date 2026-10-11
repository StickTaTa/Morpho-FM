from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from PIL import Image
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"E:\Morpho-FM")
OUT_DIR = Path(
    r"C:\Users\Administrator\Desktop\人生的第4篇论文\Figure\Figure_6"
    r"\HER2ST_patient_H\05_H1_anchored_candidate_domains"
)

SLIDES = ("H1", "H2", "H3")
DOMAIN_COLORS = {
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#2ca02c",
    3: "#d62728",
    4: "#9467bd",
    5: "#8c564b",
}
MIN_ANCHOR_SPOTS = 20

SIGNATURES = {
    "Epithelial/tumour": ["KRT19", "EPCAM", "ERBB2", "MDK"],
    "Basal/myoepithelial": ["KRT14", "KRT5", "ACTA2"],
    "Stromal": ["COL1A1", "COL3A1", "DCN", "MGP"],
    "Vascular": ["PECAM1", "VWF"],
    "Immune/plasma": ["CD74", "LST1", "IGKC", "IGHA1", "MS4A1", "IGHG1"],
    "Proliferation": ["MKI67", "TOP2A"],
}


def pred_dir(slide: str) -> Path:
    return (
        ROOT
        / "results"
        / "her2st_patient_h"
        / "morpho"
        / "single_superres"
        / f"train_{slide}_to_{slide}"
        / slide
    )


def load_common_genes(slide: str) -> list[str]:
    path = (
        ROOT
        / "checkpoints"
        / "her2st_patient_h"
        / "st_superres"
        / "single_slide"
        / f"train_{slide}_to_{slide}"
        / "common_genes.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def load_pred(slide: str) -> np.ndarray:
    return np.load(pred_dir(slide) / "pred_bag.npy").astype(np.float32)


def load_cluster_table(slide: str) -> pd.DataFrame:
    path = (
        ROOT
        / "results"
        / "her2st_patient_h"
        / "clustering"
        / f"pred_bag_train_{slide}_to_{slide}"
        / slide
        / "cluster_assignments.csv"
    )
    return pd.read_csv(path)


def load_he(slide: str) -> Image.Image:
    return Image.open(
        ROOT / "data" / "HER2ST_breast_cancer" / "raw" / "images" / "HE" / f"{slide}.jpg"
    ).convert("RGB")


def fit_h1_anchor() -> tuple[list[str], StandardScaler, PCA, dict[int, np.ndarray], dict[str, pd.DataFrame]]:
    genes = load_common_genes("H1")
    for slide in ("H2", "H3"):
        other = load_common_genes(slide)
        if other != genes:
            raise ValueError(f"Common gene order differs for {slide}.")

    h1_pred = load_pred("H1")
    h1_clusters = load_cluster_table("H1")
    labels = h1_clusters["cluster"].to_numpy(int)
    scaler = StandardScaler()
    h1_scaled = scaler.fit_transform(h1_pred)
    pca = PCA(n_components=min(30, h1_scaled.shape[0] - 1, h1_scaled.shape[1]), random_state=42)
    h1_z = pca.fit_transform(h1_scaled)

    counts = pd.Series(labels).value_counts().sort_index()
    anchor_domains = [int(d) for d, n in counts.items() if n >= MIN_ANCHOR_SPOTS]
    centroids = {}
    for domain in anchor_domains:
        centroids[int(domain)] = h1_z[labels == domain].mean(axis=0)

    domain_ids = np.array(sorted(centroids), dtype=int)
    centroid_matrix = np.vstack([centroids[i] for i in domain_ids])
    h1_nearest = pairwise_distances_argmin(h1_z, centroid_matrix, metric="cosine")
    assignments = {
        "H1": h1_clusters.assign(anchor_domain=domain_ids[h1_nearest])
    }
    return genes, scaler, pca, centroids, assignments


def project_slide(
    slide: str,
    scaler: StandardScaler,
    pca: PCA,
    centroids: dict[int, np.ndarray],
) -> pd.DataFrame:
    table = load_cluster_table(slide)
    pred = load_pred(slide)
    z = pca.transform(scaler.transform(pred))
    domain_ids = np.array(sorted(centroids), dtype=int)
    centroid_matrix = np.vstack([centroids[i] for i in domain_ids])
    nearest = pairwise_distances_argmin(z, centroid_matrix, metric="cosine")
    return table.assign(anchor_domain=domain_ids[nearest])


def panel_bounds(points: pd.DataFrame, image: Image.Image, margin: int = 360) -> tuple[int, int, int, int]:
    x0 = max(0, int(points["pixel_x"].min()) - margin)
    x1 = min(image.size[0], int(points["pixel_x"].max()) + margin)
    y0 = max(0, int(points["pixel_y"].min()) - margin)
    y1 = min(image.size[1], int(points["pixel_y"].max()) + margin)
    return x0, x1, y0, y1


def draw_background(ax, image: Image.Image, bounds: tuple[int, int, int, int]) -> None:
    x0, x1, y0, y1 = bounds
    crop = image.crop((x0, y0, x1, y1))
    ax.imshow(crop, extent=[x0, x1, y1, y0], interpolation="bilinear")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_interpolated_domains(
    ax,
    table: pd.DataFrame,
    bounds: tuple[int, int, int, int],
    scale: int = 22,
    max_distance_px: float = 185,
) -> None:
    x0, x1, y0, y1 = bounds
    xs = np.arange(x0, x1 + scale, scale)
    ys = np.arange(y0, y1 + scale, scale)
    xx, yy = np.meshgrid(xs, ys)
    points = table[["pixel_x", "pixel_y"]].to_numpy(float)
    domains = table["anchor_domain"].to_numpy(int)
    tree = cKDTree(points)
    dist, idx = tree.query(np.c_[xx.ravel(), yy.ravel()], k=1)
    grid = domains[idx].reshape(xx.shape).astype(float)
    grid[dist.reshape(xx.shape) > max_distance_px] = np.nan
    filled = np.where(np.isnan(grid), -1, grid).astype(int)
    smoothed = median_filter(filled, size=3).astype(float)
    smoothed[smoothed < 0] = np.nan

    cmap = ListedColormap([DOMAIN_COLORS[i] for i in sorted(DOMAIN_COLORS)])
    norm = BoundaryNorm(np.arange(-0.5, len(DOMAIN_COLORS) + 0.5, 1), cmap.N)
    ax.imshow(
        smoothed,
        origin="upper",
        extent=[x0, x1, y1, y0],
        cmap=cmap,
        norm=norm,
        alpha=0.30,
        interpolation="nearest",
    )
    for domain, color in DOMAIN_COLORS.items():
        arr = (smoothed == domain).astype(float)
        if np.nansum(arr) < 8:
            continue
        ax.contour(
            arr,
            levels=[0.5],
            colors=[color],
            linewidths=0.9,
            origin="upper",
            extent=[x0, x1, y1, y0],
        )


def plot_projection(assignments: dict[str, pd.DataFrame]) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(6.69, 2.45), dpi=600)
    for ax, slide in zip(axes, SLIDES):
        image = load_he(slide)
        table = assignments[slide]
        bounds = panel_bounds(table, image)
        draw_background(ax, image, bounds)
        draw_interpolated_domains(ax, table, bounds)
        ax.set_title(f"{slide} H1-anchored candidate domains", fontsize=8.5, pad=2)
        ax.text(
            0.02,
            0.95,
            slide,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1.3},
        )
    active_domains = sorted(set(np.concatenate([assignments[s]["anchor_domain"].to_numpy(int) for s in SLIDES])))
    handles = [
        Line2D([0], [0], color=DOMAIN_COLORS[i], lw=3, label=f"Domain {i}")
        for i in active_domains
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=6,
        frameon=False,
        fontsize=8,
        handlelength=1.4,
        columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.01, right=0.995, top=0.88, bottom=0.16, wspace=0.04)
    for ext in ("png", "pdf", "tif"):
        fig.savefig(OUT_DIR / f"HER2ST_H1_anchored_candidate_domain_projection.{ext}", dpi=600)
    plt.close(fig)


def zscore_by_gene(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (matrix - mean) / std


def enrichment_tables(genes: list[str], assignments: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_to_idx = {gene: i for i, gene in enumerate(genes)}
    rows = []
    sig_rows = []
    all_marker_genes = sorted({g for gs in SIGNATURES.values() for g in gs if g in gene_to_idx})

    for slide in SLIDES:
        pred = load_pred(slide)
        z = zscore_by_gene(pred)
        domains = assignments[slide]["anchor_domain"].to_numpy(int)

        for domain in sorted(np.unique(domains)):
            keep = domains == domain
            if not np.any(keep):
                continue
            for gene in all_marker_genes:
                idx = gene_to_idx[gene]
                rows.append(
                    {
                        "section": slide,
                        "domain": domain,
                        "gene": gene,
                        "mean_z": float(np.nanmean(z[keep, idx])),
                        "n_spots": int(keep.sum()),
                    }
                )
            for signature, sig_genes in SIGNATURES.items():
                idxs = [gene_to_idx[g] for g in sig_genes if g in gene_to_idx]
                if not idxs:
                    continue
                sig_rows.append(
                    {
                        "section": slide,
                        "domain": domain,
                        "signature": signature,
                        "mean_z": float(np.nanmean(z[keep][:, idxs])),
                        "genes_used": ";".join([genes[i] for i in idxs]),
                        "n_spots": int(keep.sum()),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(sig_rows)


def plot_signature_heatmap(sig_df: pd.DataFrame) -> None:
    pooled = (
        sig_df.groupby(["domain", "signature"], as_index=False)["mean_z"]
        .mean()
        .pivot(index="signature", columns="domain", values="mean_z")
        .reindex(index=list(SIGNATURES))
        .reindex(columns=sorted(sig_df["domain"].unique()))
    )
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.65, 2.55), dpi=600)
    sns.heatmap(
        pooled,
        ax=ax,
        cmap="vlag",
        center=0,
        vmin=-1.2,
        vmax=1.2,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Mean z-score", "shrink": 0.75},
    )
    ax.set_xlabel("H1-anchored candidate domain", fontsize=8.5)
    ax.set_ylabel("")
    ax.set_title("Marker/signature enrichment", fontsize=9, pad=4)
    ax.tick_params(axis="x", labelrotation=0, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf", "tif"):
        fig.savefig(OUT_DIR / f"HER2ST_H1_anchored_signature_enrichment_heatmap.{ext}", dpi=600)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    genes, scaler, pca, centroids, assignments = fit_h1_anchor()
    for slide in ("H2", "H3"):
        assignments[slide] = project_slide(slide, scaler, pca, centroids)

    combined = []
    for slide, table in assignments.items():
        out_table = table.copy()
        out_table.insert(0, "section", slide)
        combined.append(out_table)
        out_table.to_csv(OUT_DIR / f"{slide}_H1_anchored_candidate_domain_assignments.csv", index=False)
    pd.concat(combined, ignore_index=True).to_csv(
        OUT_DIR / "HER2ST_H1_anchored_candidate_domain_assignments.csv", index=False
    )

    proportions = (
        pd.concat(combined, ignore_index=True)
        .groupby(["section", "anchor_domain"])
        .size()
        .rename("n_spots")
        .reset_index()
    )
    totals = proportions.groupby("section")["n_spots"].transform("sum")
    proportions["fraction"] = proportions["n_spots"] / totals
    proportions.to_csv(OUT_DIR / "HER2ST_H1_anchored_candidate_domain_proportions.csv", index=False)

    gene_df, sig_df = enrichment_tables(genes, assignments)
    gene_df.to_csv(OUT_DIR / "HER2ST_H1_anchored_domain_marker_gene_enrichment.csv", index=False)
    sig_df.to_csv(OUT_DIR / "HER2ST_H1_anchored_domain_signature_enrichment.csv", index=False)

    plot_projection(assignments)
    plot_signature_heatmap(sig_df)

    manifest = pd.DataFrame(
        [
            {
                "file": "HER2ST_H1_anchored_candidate_domain_projection.png/pdf/tif",
                "description": "H1 predicted-expression domains projected to H1, H2 and H3.",
            },
            {
                "file": "HER2ST_H1_anchored_signature_enrichment_heatmap.png/pdf/tif",
                "description": "Pooled domain-by-signature enrichment from predicted expression.",
            },
            {
                "file": "HER2ST_H1_anchored_candidate_domain_assignments.csv",
                "description": "Spot-level projected H1-anchored domain assignment.",
            },
            {
                "file": "HER2ST_H1_anchored_domain_signature_enrichment.csv",
                "description": "Section-wise signature enrichment by H1-anchored domain.",
            },
        ]
    )
    manifest.to_csv(OUT_DIR / "HER2ST_H1_anchored_candidate_domain_manifest.csv", index=False)


if __name__ == "__main__":
    main()
