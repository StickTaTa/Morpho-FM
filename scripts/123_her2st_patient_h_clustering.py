from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("MORPHO_VC_ROOT", Path(__file__).resolve().parents[1])).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster HER2ST patient H measured or predicted spot-level expression.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--section", choices=["H1", "H2", "H3"], required=True)
    parser.add_argument("--source", choices=["measured", "pred_bag"], default="measured")
    parser.add_argument("--train-slide", choices=["H1", "H2", "H3"], default=None, help="Required for source=pred_bag.")
    parser.add_argument("--tag", default=None, help="Prediction fold tag. Defaults to train_{train-slide}_to_{section}.")
    parser.add_argument("--genes-file", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=512, help="Fallback number of variable genes when no gene list is supplied.")
    parser.add_argument("--k", type=int, default=None, help="Cluster count. Defaults to number of labeled classes on H1 or 7 otherwise.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def default_tag(train_slide: str, section: str) -> str:
    return f"train_{train_slide}_to_{section}"


def resolve_genes(adata: ad.AnnData, args: argparse.Namespace) -> list[str]:
    if args.genes_file is not None:
        genes = [g.strip() for g in args.genes_file.read_text(encoding="utf-8").splitlines() if g.strip()]
        return [g for g in genes if g in adata.var_names]

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    vars_ = X.var(axis=0)
    order = np.argsort(-vars_)
    top_n = min(args.top_n, adata.n_vars)
    return [adata.var_names[i] for i in order[:top_n]]


def load_matrix(root: Path, args: argparse.Namespace):
    adata = ad.read_h5ad(root / "data" / "spatial_data" / f"{args.section}.h5ad")
    if args.source == "measured":
        genes = resolve_genes(adata, args)
        X = adata[:, genes].X
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float32), genes, adata

    if args.train_slide is None:
        raise ValueError("--train-slide is required for source=pred_bag")
    tag = args.tag or default_tag(args.train_slide, args.section)
    pred_dir = root / "results" / "her2st_patient_h" / "morpho" / "single_superres" / tag / args.section
    pred_path = pred_dir / "pred_bag.npy"
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction not found: {pred_path}")
    genes = json.loads((root / "checkpoints" / "her2st_patient_h" / "st_superres" / "single_slide" / tag / "common_genes.json").read_text(encoding="utf-8"))
    genes = [g for g in genes if g in adata.var_names]
    pred = np.load(pred_path)
    return np.asarray(pred, dtype=np.float32), genes, adata


def choose_k(adata: ad.AnnData, args: argparse.Namespace) -> int:
    if args.k is not None:
        return args.k
    if args.section == "H1":
        labels = adata.obs["label"].astype(str)
        labels = labels[~labels.isin(["undetermined", "unlabeled"])]
        n = labels.nunique()
        if n > 1:
            return int(n)
    return 7


def cluster_matrix(X: np.ndarray, k: int, seed: int):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n_components = min(30, Xs.shape[0] - 1, Xs.shape[1])
    if n_components < 2:
        raise ValueError("Not enough dimensions to cluster.")
    pca = PCA(n_components=n_components, random_state=seed)
    Z = pca.fit_transform(Xs)
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=min(4096, len(Z)), n_init="auto")
    labels = km.fit_predict(Z)
    return Z, labels, pca


def evaluate_h1(adata: ad.AnnData, pred_labels: np.ndarray) -> dict[str, float | int | str]:
    true_labels = adata.obs["label"].astype(str).to_numpy()
    keep = ~np.isin(true_labels, ["undetermined", "unlabeled"])
    if keep.sum() == 0:
        return {"ari": np.nan, "nmi": np.nan, "n_labeled": 0}
    ari = adjusted_rand_score(true_labels[keep], pred_labels[keep])
    nmi = normalized_mutual_info_score(true_labels[keep], pred_labels[keep])
    return {"ari": float(ari), "nmi": float(nmi), "n_labeled": int(keep.sum())}


def render_spatial_plot(root: Path, adata: ad.AnnData, labels: np.ndarray, out_path: Path, title: str) -> None:
    he = np.array(Image.open(root / "data" / "superres_cache" / adata.uns["subseries"] / "he.jpg"))
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)

    fig = Figure(figsize=(6.5, 6.2))
    FigureCanvasAgg(fig)
    ax = fig.subplots(1, 1)
    ax.imshow(he)
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        s=16,
        cmap="tab20",
        alpha=0.9,
        edgecolors="none",
        rasterized=True,
    )
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    fig.clear()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    X, genes, adata = load_matrix(root, args)
    k = choose_k(adata, args)
    Z, cluster_labels, pca = cluster_matrix(X, k=k, seed=args.seed)

    tag = args.tag or (default_tag(args.train_slide, args.section) if args.train_slide else "measured")
    out_dir = root / "results" / "her2st_patient_h" / "clustering" / f"{args.source}_{tag}" / args.section
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "spot_id": adata.obs_names.astype(str),
            "pixel_x": adata.obsm["spatial"][:, 0],
            "pixel_y": adata.obsm["spatial"][:, 1],
            "cluster": cluster_labels.astype(int),
            "label": adata.obs["label"].astype(str).to_numpy(),
        }
    ).to_csv(out_dir / "cluster_assignments.csv", index=False)

    np.save(out_dir / "pca_embedding.npy", Z)
    (out_dir / "genes_used.txt").write_text("\n".join(map(str, genes)), encoding="utf-8")

    metrics = {
        "section": args.section,
        "source": args.source,
        "train_slide": args.train_slide,
        "tag": tag,
        "k": int(k),
        "n_spots": int(adata.n_obs),
        "n_genes_used": int(len(genes)),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }
    if args.section == "H1":
        metrics.update(evaluate_h1(adata, cluster_labels))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    render_spatial_plot(root, adata, cluster_labels, out_dir / "cluster_map.png", f"{args.section} | {args.source} clusters (k={k})")
    print(json.dumps(metrics, indent=2))
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
