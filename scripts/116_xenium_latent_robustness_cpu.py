from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "st_superres" / "xenium_extended" / "latent_robustness_cpu"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

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


def batch_slices(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        yield start, min(start + batch_size, n)


def load_labels() -> tuple[np.ndarray, np.ndarray]:
    ann = pd.read_excel(ROOT / "notebooks" / "breast_cancer_info.xlsx", sheet_name="Fig. 3e-j Xenium")
    fine = ann["Cluster"].astype(str).to_numpy()
    coarse = ann["Cluster"].astype(str).map(COARSE_MAP).fillna("Other").to_numpy()
    return fine, coarse


def spot_grid_feature_indices() -> tuple[np.ndarray, np.ndarray]:
    cache_dir = ROOT / "data" / "superres_cache" / "XEN1"
    locs = pd.read_csv(cache_dir / "locs.tsv", sep="\t")
    locs_grid = np.round(np.stack([locs["y"].to_numpy(), locs["x"].to_numpy()], axis=1) / 16.0).astype(np.int64)

    latent_dir = ROOT / "results" / "st_superres" / "xenium_latent_clustering" / "xenium_XEN1"
    grid_valid = np.load(latent_dir / "grid_valid_indices.npy", mmap_mode="r")
    cluster_map = np.load(latent_dir / "grid_latent_cluster_map.npy", mmap_mode="r")
    h, w = cluster_map.shape
    flat_lookup = np.full(h * w, -1, dtype=np.int64)
    flat_lookup[grid_valid[:, 0] * w + grid_valid[:, 1]] = np.arange(len(grid_valid), dtype=np.int64)
    idx = flat_lookup[locs_grid[:, 0] * w + locs_grid[:, 1]]
    keep = idx >= 0
    missing = int((~keep).sum())
    if missing:
        print(
            f"[warn] {missing:,} spot locations were not found in grid latent features; "
            "CPU robustness will use the grid-valid subset."
        )
    np.save(OUT_DIR / "xenium_xen1_grid_valid_spot_mask.npy", keep)
    np.save(OUT_DIR / "xenium_xen1_spot_grid_feature_indices.npy", idx[keep])
    return idx[keep], keep


def compute_latent_pca(batch_size: int, n_components: int) -> np.ndarray:
    out_path = OUT_DIR / f"xenium_xen1_morphofm_cellfm_latent_pca{n_components}.npy"
    if out_path.exists():
        return np.load(out_path, mmap_mode="r")

    latent_dir = ROOT / "results" / "st_superres" / "xenium_latent_clustering" / "xenium_XEN1"
    grid_features = np.load(latent_dir / "grid_latent_features.npy", mmap_mode="r")
    idx, keep = spot_grid_feature_indices()
    n = len(idx)
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    for start, end in batch_slices(n, batch_size):
        ipca.partial_fit(np.asarray(grid_features[idx[start:end]], dtype=np.float32))
        print(f"[latent PCA fit] {end:,}/{n:,}")

    scores = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n, n_components))
    for start, end in batch_slices(n, batch_size):
        scores[start:end] = ipca.transform(np.asarray(grid_features[idx[start:end]], dtype=np.float32)).astype(np.float32)
        print(f"[latent PCA transform] {end:,}/{n:,}")
    scores.flush()
    pd.DataFrame(
        {
            "component": np.arange(1, n_components + 1),
            "explained_variance_ratio": ipca.explained_variance_ratio_,
        }
    ).to_csv(OUT_DIR / f"xenium_xen1_morphofm_cellfm_latent_pca{n_components}_variance.csv", index=False)
    return np.load(out_path, mmap_mode="r")


def compute_raw_morphology_pca(batch_size: int, n_components: int) -> np.ndarray:
    out_path = OUT_DIR / f"xenium_xen1_raw_morphology_pca{n_components}.npy"
    if out_path.exists():
        return np.load(out_path, mmap_mode="r")

    cache_dir = ROOT / "data" / "superres_cache" / "XEN1"
    locs = pd.read_csv(cache_dir / "locs.tsv", sep="\t")
    keep_path = OUT_DIR / "xenium_xen1_grid_valid_spot_mask.npy"
    keep = np.load(keep_path) if keep_path.exists() else np.ones(len(locs), dtype=bool)
    locs = locs.loc[keep].reset_index(drop=True)
    locs_grid = np.round(np.stack([locs["y"].to_numpy(), locs["x"].to_numpy()], axis=1) / 16.0).astype(np.int64)
    with open(cache_dir / "embeddings-hist.pickle", "rb") as f:
        emb_dict = pickle.load(f)
    embs = np.concatenate([emb_dict["cls"], emb_dict["sub"], emb_dict["rgb"]]).transpose(1, 2, 0).astype(np.float32)
    x = embs[locs_grid[:, 0], locs_grid[:, 1]]
    x = np.asarray(x, dtype=np.float32)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    x = (x - mean) / std
    x[~np.isfinite(x)] = 0.0

    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    for start, end in batch_slices(x.shape[0], batch_size):
        ipca.partial_fit(x[start:end])
        print(f"[raw PCA fit] {end:,}/{x.shape[0]:,}")

    scores = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(x.shape[0], n_components))
    for start, end in batch_slices(x.shape[0], batch_size):
        scores[start:end] = ipca.transform(x[start:end]).astype(np.float32)
        print(f"[raw PCA transform] {end:,}/{x.shape[0]:,}")
    scores.flush()
    pd.DataFrame(
        {
            "component": np.arange(1, n_components + 1),
            "explained_variance_ratio": ipca.explained_variance_ratio_,
        }
    ).to_csv(OUT_DIR / f"xenium_xen1_raw_morphology_pca{n_components}_variance.csv", index=False)
    return np.load(out_path, mmap_mode="r")


def evaluate_feature_space(
    name: str,
    features: np.ndarray,
    fine_labels: np.ndarray,
    coarse_labels: np.ndarray,
    ks: list[int],
    seeds: list[int],
) -> pd.DataFrame:
    rows = []
    for k in ks:
        for seed in seeds:
            km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init="auto")
            labels = km.fit_predict(features)
            rows.append(
                {
                    "feature_space": name,
                    "k": k,
                    "seed": seed,
                    "fine_nmi": float(normalized_mutual_info_score(fine_labels, labels)),
                    "fine_ari": float(adjusted_rand_score(fine_labels, labels)),
                    "coarse_nmi": float(normalized_mutual_info_score(coarse_labels, labels)),
                    "coarse_ari": float(adjusted_rand_score(coarse_labels, labels)),
                }
            )
            print(f"[kmeans] {name} k={k} seed={seed}")
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["feature_space", "k"], as_index=False)
        .agg(
            fine_nmi_mean=("fine_nmi", "mean"),
            fine_nmi_sd=("fine_nmi", "std"),
            fine_ari_mean=("fine_ari", "mean"),
            fine_ari_sd=("fine_ari", "std"),
            coarse_nmi_mean=("coarse_nmi", "mean"),
            coarse_nmi_sd=("coarse_nmi", "std"),
            coarse_ari_mean=("coarse_ari", "mean"),
            coarse_ari_sd=("coarse_ari", "std"),
        )
    )


def plot_metrics(metrics: pd.DataFrame) -> None:
    long = metrics.melt(
        id_vars=["feature_space", "k", "seed"],
        value_vars=["fine_nmi", "fine_ari", "coarse_nmi", "coarse_ari"],
        var_name="metric",
        value_name="score",
    )
    metric_order = ["fine_nmi", "fine_ari", "coarse_nmi", "coarse_ari"]
    feature_spaces = list(metrics["feature_space"].drop_duplicates())
    fig, axes = plt.subplots(1, len(metric_order), figsize=(4.2 * len(metric_order), 4.5), sharey=False)
    for ax, metric in zip(axes, metric_order):
        sub = long[long["metric"] == metric]
        positions = []
        data = []
        labels = []
        pos = 1
        for k in sorted(sub["k"].unique()):
            for feature_space in feature_spaces:
                vals = sub[(sub["k"] == k) & (sub["feature_space"] == feature_space)]["score"].to_numpy()
                data.append(vals)
                positions.append(pos)
                labels.append(f"{feature_space}\nk={k}")
                pos += 1
            pos += 0.5
        ax.boxplot(data, positions=positions, widths=0.55, showfliers=False)
        ax.scatter(
            np.concatenate([np.full(len(vals), positions[i]) for i, vals in enumerate(data)]),
            np.concatenate(data),
            s=14,
            alpha=0.75,
            color="#444444",
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "xenium_xen1_latent_robustness_k_seed_metrics.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "xenium_xen1_latent_robustness_k_seed_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--n-components", type=int, default=50)
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 15, 20])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--skip-raw", action="store_true", help="Skip raw morphology feature baseline.")
    args = parser.parse_args()

    metrics_frames = []
    latent_pca = compute_latent_pca(batch_size=args.batch_size, n_components=args.n_components)
    keep = np.load(OUT_DIR / "xenium_xen1_grid_valid_spot_mask.npy")
    fine_labels, coarse_labels = load_labels()
    fine_labels = fine_labels[keep]
    coarse_labels = coarse_labels[keep]
    metrics_frames.append(
        evaluate_feature_space("morphofm_cellfm_latent", latent_pca, fine_labels, coarse_labels, args.ks, args.seeds)
    )
    if not args.skip_raw:
        raw_pca = compute_raw_morphology_pca(batch_size=args.batch_size, n_components=args.n_components)
        metrics_frames.append(
            evaluate_feature_space("raw_morphology", raw_pca, fine_labels, coarse_labels, args.ks, args.seeds)
        )

    metrics = pd.concat(metrics_frames, ignore_index=True)
    metrics.to_csv(OUT_DIR / "xenium_xen1_latent_robustness_detail.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(OUT_DIR / "xenium_xen1_latent_robustness_summary.csv", index=False)
    plot_metrics(metrics)
    print("[done] XEN1 latent robustness CPU analysis written to", OUT_DIR)


if __name__ == "__main__":
    main()
