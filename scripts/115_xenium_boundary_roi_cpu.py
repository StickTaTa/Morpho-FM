from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "st_superres" / "xenium_extended" / "boundary_roi_cpu"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

COMMON_GENES = json.loads((ROOT / "checkpoints" / "st_superres" / "xenium_cellfm" / "common_genes.json").read_text(encoding="utf-8"))
GENE_TO_IDX = {gene: idx for idx, gene in enumerate(COMMON_GENES)}

COARSE_MAP = {
    "Invasive_Tumor": "Tumor",
    "Prolif_Invasive_Tumor": "Tumor",
    "T_Cell_&_Tumor_Hybrid": "Tumor_immune_interface",
    "DCIS 1": "DCIS_myoepithelial",
    "DCIS 2": "DCIS_myoepithelial",
    "Myoepi_ACTA2+": "DCIS_myoepithelial",
    "Myoepi_KRT15+": "DCIS_myoepithelial",
    "Stromal": "Stromal_vascular",
    "Endothelial": "Stromal_vascular",
    "Perivascular-Like": "Stromal_vascular",
    "Macrophages_1": "Immune",
    "Macrophages_2": "Immune",
    "CD4+_T_Cells": "Immune",
    "CD8+_T_Cells": "Immune",
    "B_Cells": "Immune",
    "IRF7+_DCs": "Immune",
    "LAMP3+_DCs": "Immune",
    "Mast_Cells": "Immune",
    "Stromal_&_T_Cell_Hybrid": "Immune_stromal_interface",
    "Unlabeled": "Unlabeled",
}

SIGNATURES = {
    "tumour_luminal": ["ERBB2", "FOXA1", "GATA3"],
    "epithelial": ["EPCAM", "KRT7", "TACSTD2", "CEACAM6"],
    "proliferation": ["CCND1"],
    "stromal_vascular": ["ACTA2", "CAV1", "AQP1"],
    "immune": ["C1QA"],
}


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    a = a[mask] - a[mask].mean()
    b = b[mask] - b[mask].mean()
    denom = np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b))
    if denom == 0:
        return float("nan")
    return float(np.sum(a * b) / denom)


def zscore_columns(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    out = np.empty_like(arr, dtype=np.float64)
    for i in range(arr.shape[1]):
        x = arr[:, i]
        mask = np.isfinite(x)
        if mask.sum() < 3:
            out[:, i] = np.nan
            continue
        mean = x[mask].mean()
        std = x[mask].std(ddof=0)
        if std == 0 or not np.isfinite(std):
            out[:, i] = np.nan
        else:
            out[:, i] = (x - mean) / std
    return out


def load_dense_sources() -> tuple[list[str], dict[str, np.ndarray], str]:
    quant_root = ROOT / "results" / "st_superres" / "xenium_quant_summary"
    genes = json.loads((quant_root / "XEN1_within" / "selected_genes.json").read_text(encoding="utf-8"))
    sources = {
        "within_dense": np.load(quant_root / "XEN1_within" / "pred_spot_from_superres_selected.npy"),
        "cross_dense": np.load(quant_root / "XEN1_cross" / "pred_spot_from_superres_selected.npy"),
    }
    return genes, sources, "existing_12_gene_dense_backprojection"


def load_data() -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray], list[str], str]:
    ann = pd.read_excel(ROOT / "notebooks" / "breast_cancer_info.xlsx", sheet_name="Fig. 3e-j Xenium")
    ann["fine_label"] = ann["Cluster"].astype(str)
    ann["coarse_compartment"] = ann["fine_label"].map(COARSE_MAP).fillna("Other")
    coords = ann[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float64)

    dense_genes, dense_sources, source_note = load_dense_sources()
    true_all = np.load(ROOT / "results" / "st_superres" / "xenium_cellfm_1" / "xenium_XEN1" / "true_spot.npy", mmap_mode="r")
    measured = np.column_stack([true_all[:, GENE_TO_IDX[gene]] for gene in dense_genes]).astype(np.float32)
    sources = {"measured": measured, **dense_sources}
    return ann, coords, sources, dense_genes, source_note


def compute_signed_tumour_distance(ann: pd.DataFrame, coords: np.ndarray) -> np.ndarray:
    tumour = ann["coarse_compartment"].to_numpy() == "Tumor"
    non_tumour = ~tumour
    valid = np.isfinite(coords).all(axis=1)
    tumour &= valid
    non_tumour &= valid
    if tumour.sum() == 0 or non_tumour.sum() == 0:
        raise RuntimeError("Need both tumour and non-tumour cells to compute boundary distances.")
    tumour_tree = cKDTree(coords[tumour])
    non_tumour_tree = cKDTree(coords[non_tumour])
    dist = np.full(len(coords), np.nan, dtype=np.float64)
    dist[tumour] = non_tumour_tree.query(coords[tumour], k=1)[0]
    dist[non_tumour] = -tumour_tree.query(coords[non_tumour], k=1)[0]
    return dist


def source_signature_scores(sources: dict[str, np.ndarray], genes: list[str]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    gene_pos = {gene: i for i, gene in enumerate(genes)}
    rows = []
    score_tables: dict[str, pd.DataFrame] = {}
    for source_name, values in sources.items():
        z = zscore_columns(values)
        sig_cols = {}
        for signature, sig_genes in SIGNATURES.items():
            available = [gene for gene in sig_genes if gene in gene_pos]
            if not available:
                continue
            idx = [gene_pos[gene] for gene in available]
            sig_cols[signature] = np.nanmean(z[:, idx], axis=1)
            rows.append({"source": source_name, "signature": signature, "genes": ";".join(available), "n_genes": len(available)})
        score_tables[source_name] = pd.DataFrame(sig_cols)
    return score_tables, pd.DataFrame(rows)


def boundary_gradient(ann: pd.DataFrame, signed_distance: np.ndarray, scores: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    bins = np.array([-400, -250, -150, -100, -50, -25, 0, 25, 50, 100, 150, 250, 400], dtype=float)
    bin_ids = pd.cut(signed_distance, bins=bins, include_lowest=True)
    gradient_rows = []
    for source, df in scores.items():
        for signature in df.columns:
            vals = df[signature].to_numpy()
            for interval in bin_ids.categories:
                mask = np.asarray(bin_ids == interval) & np.isfinite(vals)
                if mask.sum() == 0:
                    mean = sem = np.nan
                else:
                    mean = float(vals[mask].mean())
                    sem = float(vals[mask].std(ddof=0) / np.sqrt(mask.sum()))
                gradient_rows.append(
                    {
                        "source": source,
                        "signature": signature,
                        "distance_bin": str(interval),
                        "distance_mid_um": float((interval.left + interval.right) / 2),
                        "n_cells": int(mask.sum()),
                        "mean_zscore": mean,
                        "sem_zscore": sem,
                    }
                )
    gradient_df = pd.DataFrame(gradient_rows)

    agreement_rows = []
    measured = gradient_df[gradient_df["source"] == "measured"]
    for source in [name for name in scores if name != "measured"]:
        pred = gradient_df[gradient_df["source"] == source]
        for signature in measured["signature"].unique():
            m = measured[measured["signature"] == signature].sort_values("distance_mid_um")
            p = pred[pred["signature"] == signature].sort_values("distance_mid_um")
            merged = m[["distance_mid_um", "mean_zscore"]].merge(
                p[["distance_mid_um", "mean_zscore"]],
                on="distance_mid_um",
                suffixes=("_measured", "_predicted"),
            )
            r = pearson_corr(merged["mean_zscore_measured"].to_numpy(), merged["mean_zscore_predicted"].to_numpy())
            mae = np.nanmean(np.abs(merged["mean_zscore_measured"] - merged["mean_zscore_predicted"]))
            agreement_rows.append({"source": source, "signature": signature, "gradient_pearson": r, "gradient_mae": float(mae)})
    return gradient_df, pd.DataFrame(agreement_rows)


def make_roi_masks(ann: pd.DataFrame, signed_distance: np.ndarray, sources: dict[str, np.ndarray], genes: list[str]) -> dict[str, np.ndarray]:
    coarse = ann["coarse_compartment"].to_numpy()
    erbb2_idx = genes.index("ERBB2") if "ERBB2" in genes else None
    roi_masks: dict[str, np.ndarray] = {
        "tumour_core": (coarse == "Tumor") & (signed_distance >= 100),
        "tumour_boundary_band": np.abs(signed_distance) <= 50,
        "adjacent_stroma": (coarse == "Stromal_vascular") & (signed_distance >= -150) & (signed_distance < 0),
        "dcis_myoepithelial": coarse == "DCIS_myoepithelial",
        "stromal_vascular": coarse == "Stromal_vascular",
        "immune_enriched": coarse == "Immune",
    }
    if erbb2_idx is not None:
        measured_erbb2 = sources["measured"][:, erbb2_idx]
        tumour = coarse == "Tumor"
        cutoff = np.nanpercentile(measured_erbb2[tumour], 75)
        roi_masks["erbb2_high_tumour"] = tumour & (measured_erbb2 >= cutoff)

    try:
        roi_df = pd.read_excel(ROOT / "notebooks" / "breast_cancer_info.xlsx", sheet_name="Fig. 4b-d ")
        barcode_to_index = pd.Series(np.arange(len(ann)), index=ann["Barcode"].astype(int)).to_dict()
        for roi_name, sub in roi_df.groupby("ROI_Membership"):
            if pd.isna(roi_name):
                continue
            idx = [barcode_to_index.get(int(bc)) for bc in sub["Barcode"].dropna()]
            idx = np.array([i for i in idx if i is not None], dtype=int)
            if idx.size < 50:
                continue
            mask = np.zeros(len(ann), dtype=bool)
            mask[idx] = True
            roi_masks[f"named_{str(roi_name).replace(' ', '_')}"] = mask
    except Exception as exc:
        print(f"[warn] Could not add named ROI memberships: {exc}")
    return roi_masks


def roi_enrichment(scores: dict[str, pd.DataFrame], roi_masks: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for roi_name, roi_mask in roi_masks.items():
        if roi_mask.sum() < 20:
            continue
        bg_mask = ~roi_mask
        for source, df in scores.items():
            for signature in df.columns:
                vals = df[signature].to_numpy()
                roi_vals = vals[roi_mask & np.isfinite(vals)]
                bg_vals = vals[bg_mask & np.isfinite(vals)]
                if roi_vals.size == 0 or bg_vals.size == 0:
                    continue
                rows.append(
                    {
                        "roi": roi_name,
                        "source": source,
                        "signature": signature,
                        "n_roi_cells": int(roi_vals.size),
                        "roi_mean_zscore": float(roi_vals.mean()),
                        "background_mean_zscore": float(bg_vals.mean()),
                        "enrichment_delta": float(roi_vals.mean() - bg_vals.mean()),
                    }
                )
    enrich_df = pd.DataFrame(rows)
    corr_rows = []
    measured = enrich_df[enrich_df["source"] == "measured"]
    for source in sorted(set(enrich_df["source"]) - {"measured"}):
        pred = enrich_df[enrich_df["source"] == source]
        merged = measured[["roi", "signature", "enrichment_delta"]].merge(
            pred[["roi", "signature", "enrichment_delta"]],
            on=["roi", "signature"],
            suffixes=("_measured", "_predicted"),
        )
        corr_rows.append(
            {
                "source": source,
                "n_roi_signature_pairs": int(len(merged)),
                "roi_enrichment_pearson": pearson_corr(
                    merged["enrichment_delta_measured"].to_numpy(),
                    merged["enrichment_delta_predicted"].to_numpy(),
                ),
                "roi_enrichment_mae": float(
                    np.nanmean(np.abs(merged["enrichment_delta_measured"] - merged["enrichment_delta_predicted"]))
                ),
            }
        )
    return enrich_df, pd.DataFrame(corr_rows)


def plot_boundary_gradient(gradient_df: pd.DataFrame) -> None:
    signatures = [sig for sig in SIGNATURES if sig in gradient_df["signature"].unique()]
    colors = {"measured": "black", "within_dense": "#2878b5", "cross_dense": "#c75d2c"}
    fig, axes = plt.subplots(1, len(signatures), figsize=(4.0 * len(signatures), 4.2), sharey=True)
    if len(signatures) == 1:
        axes = [axes]
    for ax, signature in zip(axes, signatures):
        sub = gradient_df[gradient_df["signature"] == signature]
        for source, sdf in sub.groupby("source", sort=False):
            sdf = sdf.sort_values("distance_mid_um")
            ax.plot(sdf["distance_mid_um"], sdf["mean_zscore"], marker="o", linewidth=1.8, markersize=3, label=source, color=colors.get(source))
            ax.fill_between(
                sdf["distance_mid_um"].to_numpy(dtype=float),
                (sdf["mean_zscore"] - sdf["sem_zscore"]).to_numpy(dtype=float),
                (sdf["mean_zscore"] + sdf["sem_zscore"]).to_numpy(dtype=float),
                color=colors.get(source),
                alpha=0.12,
                linewidth=0,
            )
        ax.axvline(0, color="#666666", linewidth=0.9, linestyle="--")
        ax.set_title(signature)
        ax.set_xlabel("Signed distance to tumour edge (um)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean signature z-score")
    axes[-1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "xenium_xen1_boundary_gradient_signatures.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "xenium_xen1_boundary_gradient_signatures.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_roi_enrichment(enrich_df: pd.DataFrame) -> None:
    core_rois = [
        "erbb2_high_tumour",
        "tumour_core",
        "tumour_boundary_band",
        "adjacent_stroma",
        "dcis_myoepithelial",
        "stromal_vascular",
        "immune_enriched",
    ]
    core = enrich_df[enrich_df["roi"].isin(core_rois)].copy()
    if core.empty:
        return
    signatures = [sig for sig in SIGNATURES if sig in core["signature"].unique()]
    sources = ["measured", "within_dense", "cross_dense"]
    fig, axes = plt.subplots(1, len(sources), figsize=(4.5 * len(sources), 5.2), sharey=True)
    for ax, source in zip(axes, sources):
        pivot = core[core["source"] == source].pivot(index="roi", columns="signature", values="enrichment_delta")
        pivot = pivot.reindex(index=[roi for roi in core_rois if roi in pivot.index], columns=signatures)
        vmax = np.nanmax(np.abs(pivot.to_numpy()))
        vmax = max(vmax, 0.5)
        im = ax.imshow(pivot.to_numpy(), cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(source)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="ROI enrichment delta")
    fig.savefig(FIG_DIR / "xenium_xen1_roi_signature_enrichment_heatmap.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "xenium_xen1_roi_signature_enrichment_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ann, coords, sources, genes, source_note = load_data()
    signed_distance = compute_signed_tumour_distance(ann, coords)
    ann_out = ann[["Barcode", "fine_label", "coarse_compartment", "x_centroid", "y_centroid"]].copy()
    ann_out["signed_distance_to_tumour_edge_um"] = signed_distance
    ann_out.to_csv(OUT_DIR / "xenium_xen1_annotation_with_boundary_distance.csv", index=False)

    scores, signature_manifest = source_signature_scores(sources, genes)
    signature_manifest["source_note"] = source_note
    signature_manifest.to_csv(OUT_DIR / "xenium_xen1_signature_manifest.csv", index=False)

    gradient_df, gradient_agreement = boundary_gradient(ann, signed_distance, scores)
    gradient_df.to_csv(OUT_DIR / "xenium_xen1_boundary_gradient_bins.csv", index=False)
    gradient_agreement.to_csv(OUT_DIR / "xenium_xen1_boundary_gradient_agreement.csv", index=False)
    plot_boundary_gradient(gradient_df)

    roi_masks = make_roi_masks(ann, signed_distance, sources, genes)
    roi_summary = pd.DataFrame(
        [{"roi": name, "n_cells": int(mask.sum())} for name, mask in roi_masks.items()]
    ).sort_values("roi")
    roi_summary.to_csv(OUT_DIR / "xenium_xen1_roi_definitions.csv", index=False)
    enrich_df, enrich_corr = roi_enrichment(scores, roi_masks)
    enrich_df.to_csv(OUT_DIR / "xenium_xen1_roi_signature_enrichment.csv", index=False)
    enrich_corr.to_csv(OUT_DIR / "xenium_xen1_roi_signature_enrichment_agreement.csv", index=False)
    plot_roi_enrichment(enrich_df)
    print("[done] XEN1 boundary/ROI CPU analysis written to", OUT_DIR)


if __name__ == "__main__":
    main()
