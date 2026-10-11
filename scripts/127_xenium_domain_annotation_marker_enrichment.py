from __future__ import annotations

import math
import os
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("MORPHO_VC_ROOT", r"E:\Morpho-FM")).resolve()
sys.path.append(str(ROOT / "src"))

OUT_DIR = ROOT / "results" / "st_superres" / "xenium_latent_clustering" / "domain_biology_enrichment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SECTION_CONFIG = {
    "XEN1": {
        "sheet": "Fig. 3e-j Xenium",
        "label_col": "Cluster",
    },
    "XEN2": {
        "sheet": "Sup Fig. 5 Xenium Rep 2",
        "label_col": "Celltype",
    },
}

LABEL_ORDER = [
    "Stromal",
    "Invasive_Tumor",
    "DCIS 1",
    "DCIS 2",
    "Macrophages_1",
    "Endothelial",
    "Unlabeled",
    "CD4+_T_Cells",
    "Myoepi_ACTA2+",
    "CD8+_T_Cells",
    "B_Cells",
    "Prolif_Invasive_Tumor",
    "Myoepi_KRT15+",
    "Macrophages_2",
    "Perivascular-Like",
    "Stromal_&_T_Cell_Hybrid",
    "T_Cell_&_Tumor_Hybrid",
    "IRF7+_DCs",
    "LAMP3+_DCs",
    "Mast_Cells",
]

MYOEPITHELIAL_LABELS = {"DCIS 1", "DCIS 2", "Myoepi_ACTA2+", "Myoepi_KRT15+"}
TUMOUR_LABELS = {"Invasive_Tumor", "Prolif_Invasive_Tumor", "T_Cell_&_Tumor_Hybrid"}
STROMAL_LABELS = {"Stromal", "Endothelial", "Perivascular-Like", "Stromal_&_T_Cell_Hybrid"}
IMMUNE_LABELS = {
    "Macrophages_1",
    "Macrophages_2",
    "CD4+_T_Cells",
    "CD8+_T_Cells",
    "B_Cells",
    "IRF7+_DCs",
    "LAMP3+_DCs",
    "Mast_Cells",
}


def get_disk_mask(radius: float) -> np.ndarray:
    radius_ceil = int(np.ceil(radius))
    locs = np.meshgrid(
        np.arange(-radius_ceil, radius_ceil + 1),
        np.arange(-radius_ceil, radius_ceil + 1),
        indexing="ij",
    )
    locs = np.stack(locs, axis=-1)
    return (locs**2).sum(-1) <= radius**2


def compute_valid_indices(section: str, n_locs: int, n_labels: int) -> np.ndarray:
    if n_locs == n_labels:
        return np.arange(n_locs, dtype=np.int64)

    cache_dir = ROOT / "data" / "superres_cache" / section
    locs = pd.read_csv(cache_dir / "locs.tsv", sep="\t")
    locs_px = np.stack([locs["y"].to_numpy(), locs["x"].to_numpy()], axis=-1).astype(np.float32)
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


def load_section(section: str) -> dict[str, np.ndarray]:
    cfg = SECTION_CONFIG[section]
    cluster_dir = ROOT / "results" / "st_superres" / "xenium_latent_clustering" / f"xenium_{section}"
    labels = np.load(cluster_dir / "latent_cluster_labels.npy").astype(int)

    cache_dir = ROOT / "data" / "superres_cache" / section
    locs = pd.read_csv(cache_dir / "locs.tsv", sep="\t")
    valid_idx = compute_valid_indices(section, len(locs), len(labels))

    ann = pd.read_excel(ROOT / "notebooks" / "breast_cancer_info.xlsx", sheet_name=cfg["sheet"])
    if len(ann) != len(locs):
        raise ValueError(f"{section}: annotation rows {len(ann):,} != loc rows {len(locs):,}")
    annotation = ann[cfg["label_col"]].astype(str).to_numpy()[valid_idx]

    cnts = pd.read_csv(cache_dir / "cnts.tsv", sep="\t", usecols=["KRT14", "KRT5"])
    krt14 = cnts["KRT14"].to_numpy(dtype=np.float32)[valid_idx]
    krt5 = cnts["KRT5"].to_numpy(dtype=np.float32)[valid_idx]

    return {
        "labels": labels,
        "annotation": annotation,
        "KRT14": krt14,
        "KRT5": krt5,
    }


def safe_ratio(numer: float, denom: float) -> float:
    if denom <= 0:
        return float("nan")
    return float(numer / denom)


def annotation_enrichment(section: str, labels: np.ndarray, annotation: np.ndarray) -> pd.DataFrame:
    rows = []
    all_labels = [label for label in LABEL_ORDER if label in set(annotation)]
    all_labels += sorted(set(annotation) - set(all_labels))
    n_total = len(labels)
    domains = sorted(np.unique(labels).astype(int).tolist())
    for domain in domains:
        domain_mask = labels == domain
        n_domain = int(domain_mask.sum())
        for ann_label in all_labels:
            label_mask = annotation == ann_label
            n_label = int(label_mask.sum())
            overlap = int((domain_mask & label_mask).sum())
            frac_domain = safe_ratio(overlap, n_domain)
            frac_all = safe_ratio(n_label, n_total)
            enrichment = safe_ratio(frac_domain + 1e-9, frac_all + 1e-9)
            rows.append(
                {
                    "section": section,
                    "domain": domain,
                    "annotation_label": ann_label,
                    "n_domain_cells": n_domain,
                    "n_label_cells": n_label,
                    "n_overlap": overlap,
                    "fraction_of_domain": frac_domain,
                    "fraction_of_all_cells": frac_all,
                    "enrichment_ratio": enrichment,
                    "log2_enrichment": math.log2(enrichment) if np.isfinite(enrichment) and enrichment > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def marker_enrichment(section: str, labels: np.ndarray, marker_values: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    domains = sorted(np.unique(labels).astype(int).tolist())
    for gene, values in marker_values.items():
        log_values = np.log1p(values.astype(np.float64))
        positive = values > 0
        all_pos = float(positive.mean())
        all_mean_log = float(log_values.mean())
        for domain in domains:
            mask = labels == domain
            n_domain = int(mask.sum())
            pos_frac = float(positive[mask].mean())
            mean_log = float(log_values[mask].mean())
            rows.append(
                {
                    "section": section,
                    "domain": domain,
                    "gene": gene,
                    "n_domain_cells": n_domain,
                    "positive_fraction": pos_frac,
                    "background_positive_fraction": all_pos,
                    "positive_fraction_enrichment": safe_ratio(pos_frac + 1e-9, all_pos + 1e-9),
                    "mean_log1p_count": mean_log,
                    "background_mean_log1p_count": all_mean_log,
                    "mean_log1p_enrichment_delta": mean_log - all_mean_log,
                }
            )
    return pd.DataFrame(rows)


def domain_summary(
    section: str,
    labels: np.ndarray,
    annotation: np.ndarray,
    ann_df: pd.DataFrame,
    marker_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    domains = sorted(np.unique(labels).astype(int).tolist())
    for domain in domains:
        domain_ann = ann_df[(ann_df["section"] == section) & (ann_df["domain"] == domain)]
        top = domain_ann.sort_values(["fraction_of_domain", "enrichment_ratio"], ascending=False).iloc[0]
        top_enriched = domain_ann[domain_ann["n_overlap"] >= 50].sort_values("enrichment_ratio", ascending=False).iloc[0]

        mask = labels == domain
        ann_values = annotation[mask]
        myo_frac = float(np.isin(ann_values, list(MYOEPITHELIAL_LABELS)).mean())
        tumor_frac = float(np.isin(ann_values, list(TUMOUR_LABELS)).mean())
        stromal_frac = float(np.isin(ann_values, list(STROMAL_LABELS)).mean())
        immune_frac = float(np.isin(ann_values, list(IMMUNE_LABELS)).mean())
        marker_sub = marker_df[(marker_df["section"] == section) & (marker_df["domain"] == domain)]
        krt14 = marker_sub[marker_sub["gene"] == "KRT14"].iloc[0]
        krt5 = marker_sub[marker_sub["gene"] == "KRT5"].iloc[0]
        rows.append(
            {
                "section": section,
                "domain": domain,
                "n_domain_cells": int(mask.sum()),
                "dominant_annotation_label": top["annotation_label"],
                "dominant_annotation_fraction": float(top["fraction_of_domain"]),
                "dominant_annotation_enrichment": float(top["enrichment_ratio"]),
                "top_enriched_annotation_label_min50": top_enriched["annotation_label"],
                "top_enriched_annotation_ratio_min50": float(top_enriched["enrichment_ratio"]),
                "dcis_myoepithelial_fraction": myo_frac,
                "tumour_fraction": tumor_frac,
                "stromal_vascular_fraction": stromal_frac,
                "immune_fraction": immune_frac,
                "KRT14_positive_fraction": float(krt14["positive_fraction"]),
                "KRT14_positive_enrichment": float(krt14["positive_fraction_enrichment"]),
                "KRT14_mean_log1p_delta": float(krt14["mean_log1p_enrichment_delta"]),
                "KRT5_positive_fraction": float(krt5["positive_fraction"]),
                "KRT5_positive_enrichment": float(krt5["positive_fraction_enrichment"]),
                "KRT5_mean_log1p_delta": float(krt5["mean_log1p_enrichment_delta"]),
            }
        )
    return pd.DataFrame(rows)


def plot_annotation_heatmap(ann_df: pd.DataFrame) -> None:
    labels = [label for label in LABEL_ORDER if label in set(ann_df["annotation_label"])]
    labels += sorted(set(ann_df["annotation_label"]) - set(labels))
    for section, sub in ann_df.groupby("section"):
        pivot = sub.pivot(index="domain", columns="annotation_label", values="log2_enrichment").reindex(columns=labels)
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        im = ax.imshow(np.clip(pivot.to_numpy(dtype=float), -2, 2), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"Domain {d}" for d in pivot.index], fontsize=8)
        ax.set_title(f"{section} latent-domain annotation enrichment", fontsize=10)
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("log2 enrichment", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{section}_annotation_log2_enrichment_heatmap.pdf", bbox_inches="tight")
        fig.savefig(OUT_DIR / f"{section}_annotation_log2_enrichment_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_marker_heatmap(marker_df: pd.DataFrame) -> None:
    for section, sub in marker_df.groupby("section"):
        pivot = sub.pivot(index="domain", columns="gene", values="mean_log1p_enrichment_delta").reindex(columns=["KRT14", "KRT5"])
        fig, ax = plt.subplots(figsize=(3.0, 4.8))
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="magma_r")
        ax.set_xticks(np.arange(2))
        ax.set_xticklabels(["KRT14", "KRT5"], fontsize=8)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"Domain {d}" for d in pivot.index], fontsize=8)
        ax.set_title(f"{section} KRT14/KRT5 domain enrichment", fontsize=10)
        cbar = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.04)
        cbar.set_label("mean log1p(count) delta", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{section}_KRT14_KRT5_marker_enrichment_heatmap.pdf", bbox_inches="tight")
        fig.savefig(OUT_DIR / f"{section}_KRT14_KRT5_marker_enrichment_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ann_frames = []
    marker_frames = []
    summary_frames = []
    for section in ["XEN1", "XEN2"]:
        data = load_section(section)
        labels = data["labels"]
        annotation = data["annotation"]
        ann_df = annotation_enrichment(section, labels, annotation)
        marker_df = marker_enrichment(section, labels, {"KRT14": data["KRT14"], "KRT5": data["KRT5"]})
        summary_df = domain_summary(section, labels, annotation, ann_df, marker_df)
        ann_frames.append(ann_df)
        marker_frames.append(marker_df)
        summary_frames.append(summary_df)

    ann_all = pd.concat(ann_frames, ignore_index=True)
    marker_all = pd.concat(marker_frames, ignore_index=True)
    summary_all = pd.concat(summary_frames, ignore_index=True)

    ann_all.to_csv(OUT_DIR / "xenium_domain_annotation_enrichment.csv", index=False)
    marker_all.to_csv(OUT_DIR / "xenium_domain_KRT14_KRT5_marker_enrichment.csv", index=False)
    summary_all.to_csv(OUT_DIR / "xenium_domain_biology_summary.csv", index=False)

    myo = summary_all.sort_values(
        ["section", "dcis_myoepithelial_fraction", "KRT14_mean_log1p_delta", "KRT5_mean_log1p_delta"],
        ascending=[True, False, False, False],
    )
    myo.to_csv(OUT_DIR / "xenium_domain_dcis_myoepithelial_marker_ranked.csv", index=False)

    plot_annotation_heatmap(ann_all)
    plot_marker_heatmap(marker_all)

    print("[summary] top DCIS/myoepithelial-marker domains")
    cols = [
        "section",
        "domain",
        "n_domain_cells",
        "dominant_annotation_label",
        "dominant_annotation_fraction",
        "dcis_myoepithelial_fraction",
        "KRT14_positive_fraction",
        "KRT14_positive_enrichment",
        "KRT5_positive_fraction",
        "KRT5_positive_enrichment",
    ]
    print(myo.groupby("section", group_keys=False).head(5)[cols].to_string(index=False))
    print(f"[done] {OUT_DIR}")


if __name__ == "__main__":
    main()
