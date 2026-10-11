from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FOLD = "loso_train_H2_H3_to_H1"
LABEL_COLORS = {
    "invasive cancer": "#e41a1c",
    "cancer in situ": "#ff7f00",
    "immune infiltrate": "#ffff33",
    "breast glands": "#1fc44a",
    "adipose tissue": "#66d9d9",
    "connective tissue": "#4053c7",
    "undetermined": "#cfcfcf",
}
Image.MAX_IMAGE_PIXELS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot H1 leave-one-section-out HER2ST transfer summary.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--fold", default=FOLD)
    parser.add_argument("--section", default="H1")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--top-n", type=int, default=1000)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "results" / "her2st_patient_h" / "leave_one_section_out_figures",
    )
    parser.add_argument("--dpi", type=int, default=900)
    return parser.parse_args()


def load_inputs(root: Path, fold: str, section: str):
    result_dir = root / "results" / "her2st_patient_h" / "morpho" / "leave_one_section_out" / fold / section
    ckpt_dir = root / "checkpoints" / "her2st_patient_h" / "st_superres" / "leave_one_section_out" / fold
    adata = ad.read_h5ad(root / "data" / "spatial_data" / f"{section}.h5ad")
    pred = np.load(result_dir / "pred_bag.npy")
    true = np.load(result_dir / "true_bag.npy")
    spot_ids = [line.strip() for line in (result_dir / "spot_ids.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    genes = json.loads((ckpt_dir / "common_genes.json").read_text(encoding="utf-8"))
    metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    corr_all = np.load(result_dir / "gene_corr_all.npy")
    corr_hvg = np.load(result_dir / "gene_corr_hvg.npy")
    return result_dir, adata, pred, true, spot_ids, genes, metrics, corr_all, corr_hvg


def cluster_matrix(X: np.ndarray, k: int, seed: int = 42, max_iter: int = 100):
    X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(X.shape[0], size=k, replace=False)].copy()
    labels = np.zeros(X.shape[0], dtype=np.int32)
    for _ in range(max_iter):
        dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            mask = labels == i
            if mask.any():
                centers[i] = X[mask].mean(axis=0)
            else:
                centers[i] = X[rng.integers(0, X.shape[0])]
    return labels.astype(int)


def _comb2(x: np.ndarray) -> np.ndarray:
    return x * (x - 1) / 2.0


def _contingency(labels_true: np.ndarray, labels_pred: np.ndarray) -> np.ndarray:
    true_vals, true_inv = np.unique(labels_true, return_inverse=True)
    pred_vals, pred_inv = np.unique(labels_pred, return_inverse=True)
    table = np.zeros((len(true_vals), len(pred_vals)), dtype=np.float64)
    np.add.at(table, (true_inv, pred_inv), 1)
    return table


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    table = _contingency(labels_true, labels_pred)
    n = table.sum()
    if n < 2:
        return float("nan")
    sum_comb = _comb2(table).sum()
    sum_rows = _comb2(table.sum(axis=1)).sum()
    sum_cols = _comb2(table.sum(axis=0)).sum()
    total = _comb2(np.asarray([n]))[0]
    expected = sum_rows * sum_cols / total if total > 0 else 0.0
    max_index = 0.5 * (sum_rows + sum_cols)
    denom = max_index - expected
    return float((sum_comb - expected) / denom) if denom != 0 else 0.0


def normalized_mutual_info(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    table = _contingency(labels_true, labels_pred)
    n = table.sum()
    if n <= 0:
        return float("nan")
    pxy = table / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    mi = float((pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])).sum())
    hx = float(-(px[px > 0] * np.log(px[px > 0])).sum())
    hy = float(-(py[py > 0] * np.log(py[py > 0])).sum())
    denom = np.sqrt(hx * hy)
    return float(mi / denom) if denom > 0 else 0.0


def eval_labels(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict[str, float | int]:
    keep = ~np.isin(true_labels.astype(str), ["undetermined", "unlabeled"])
    return {
        "ari": adjusted_rand_index(true_labels[keep], pred_labels[keep]),
        "nmi": normalized_mutual_info(true_labels[keep], pred_labels[keep]),
        "n_labeled": int(keep.sum()),
    }


def scatter_label_map(ax, he, coords, labels, title: str, s: float = 13.0):
    ax.imshow(he)
    for lab in LABEL_COLORS:
        mask = labels == lab
        if mask.any():
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=s,
                c=LABEL_COLORS[lab],
                edgecolors="none",
                alpha=0.88 if lab != "undetermined" else 0.45,
                rasterized=True,
            )
    ax.set_title(title, fontsize=9, pad=3)
    ax.axis("off")


def scatter_cluster_map(ax, he, coords, clusters, title: str, k: int, s: float = 13.0):
    ax.imshow(he)
    cmap = plt.get_cmap("tab10", k)
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=s,
        c=clusters,
        cmap=cmap,
        vmin=-0.5,
        vmax=k - 0.5,
        edgecolors="none",
        alpha=0.88,
        rasterized=True,
    )
    ax.set_title(title, fontsize=9, pad=3)
    ax.axis("off")


def plot_summary(
    out_dir: Path,
    section: str,
    he,
    coords,
    labels,
    measured_clusters,
    predicted_clusters,
    measured_eval,
    predicted_eval,
    corr_all,
    corr_hvg,
    metrics,
    k: int,
    dpi: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6.69, 5.25))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.82], hspace=0.30, wspace=0.06)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_box = fig.add_subplot(gs[1, 0])
    ax_leg1 = fig.add_subplot(gs[1, 1])
    ax_leg2 = fig.add_subplot(gs[1, 2])

    scatter_label_map(axes[0], he, coords, labels, "Manual labels")
    scatter_cluster_map(
        axes[1],
        he,
        coords,
        measured_clusters,
        f"Measured-expression clusters\nARI={measured_eval['ari']:.3f}, NMI={measured_eval['nmi']:.3f}",
        k,
    )
    scatter_cluster_map(
        axes[2],
        he,
        coords,
        predicted_clusters,
        f"Predicted-expression clusters\nARI={predicted_eval['ari']:.3f}, NMI={predicted_eval['nmi']:.3f}",
        k,
    )

    data = [corr_all[np.isfinite(corr_all)], corr_hvg[np.isfinite(corr_hvg)]]
    bp = ax_box.boxplot(
        data,
        labels=["All genes", "Top HVGs"],
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "0.15", "linewidth": 1.0},
        boxprops={"edgecolor": "0.25", "linewidth": 0.8},
        whiskerprops={"color": "0.35", "linewidth": 0.8},
        capprops={"color": "0.35", "linewidth": 0.8},
    )
    for patch, color in zip(bp["boxes"], ["#b8d7f0", "#f2c36b"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.95)
    rng = np.random.default_rng(42)
    for i, arr in enumerate(data, start=1):
        if arr.size > 0:
            sample = arr if arr.size <= 800 else rng.choice(arr, size=800, replace=False)
            ax_box.scatter(
                rng.normal(i, 0.035, size=sample.size),
                sample,
                s=3,
                c="0.25",
                alpha=0.22,
                linewidths=0,
                rasterized=True,
            )
    ax_box.axhline(0, color="0.7", linewidth=0.7)
    ax_box.set_ylim(-0.25, 1.0)
    ax_box.set_ylabel("Pearson r", fontsize=8)
    ax_box.set_title("Held-out H1 prediction", fontsize=9, pad=3)
    ax_box.tick_params(axis="both", labelsize=7)
    ax_box.grid(axis="y", color="0.88", linewidth=0.6)
    ax_box.text(
        0.02,
        0.98,
        (
            f"Mean r={metrics['Mean_Pearson']:.3f}\n"
            f"HVG1000 mean r={metrics['HVG1000_Mean_Pearson']:.3f}\n"
            f"Best: {metrics['Best_Gene']} r={metrics['Best_Pearson']:.3f}"
        ),
        transform=ax_box.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )

    ax_leg1.axis("off")
    label_handles = [mpatches.Patch(color=LABEL_COLORS[lab], label=lab) for lab in LABEL_COLORS]
    ax_leg1.legend(
        handles=label_handles,
        loc="center left",
        fontsize=7,
        frameon=False,
        title="Manual labels",
        title_fontsize=8,
    )

    ax_leg2.axis("off")
    cmap = plt.get_cmap("tab10", k)
    cluster_handles = [mpatches.Patch(color=cmap(i), label=f"Cluster {i}") for i in range(k)]
    ax_leg2.legend(
        handles=cluster_handles,
        loc="center left",
        fontsize=7,
        frameon=False,
        title=f"Expression clusters (k={k})",
        title_fontsize=8,
        ncol=1,
    )
    fig.suptitle(f"{section} held-out transfer: train H2+H3, predict H1", fontsize=10, y=0.98)
    fig.savefig(out_dir / f"{section}_loso_H2_H3_to_H1_summary.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{section}_loso_H2_H3_to_H1_summary.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{section}_loso_H2_H3_to_H1_summary.tif", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_display_he(path: Path, max_dim: int = 4800) -> tuple[np.ndarray, float, float]:
    img = Image.open(path)
    orig_w, orig_h = img.size
    scale = min(1.0, float(max_dim) / float(max(orig_w, orig_h)))
    if scale < 1.0:
        new_size = (max(1, int(round(orig_w * scale))), max(1, int(round(orig_h * scale))))
        img = img.resize(new_size, resample=Image.Resampling.LANCZOS)
    arr = np.array(img)
    sx = arr.shape[1] / float(orig_w)
    sy = arr.shape[0] / float(orig_h)
    return arr, sx, sy


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    result_dir, adata, pred, true, spot_ids, genes, metrics, corr_all, corr_hvg = load_inputs(root, args.fold, args.section)
    adata_sub = adata[spot_ids].copy()
    coords = np.asarray(adata_sub.obsm["spatial"], dtype=np.float32)
    labels = adata_sub.obs["label"].astype(str).to_numpy()
    he, sx, sy = load_display_he(root / "data" / "superres_cache" / args.section / "he.jpg")
    coords = coords.copy()
    coords[:, 0] *= sx
    coords[:, 1] *= sy

    # Cluster on the top measured-variable genes, using the same gene subset for measured and predicted profiles.
    vars_ = np.var(true, axis=0)
    top_idx = np.argsort(-vars_)[: min(args.top_n, true.shape[1])]
    measured_clusters = cluster_matrix(true[:, top_idx], args.k)
    predicted_clusters = cluster_matrix(pred[:, top_idx], args.k)
    measured_eval = eval_labels(labels, measured_clusters)
    predicted_eval = eval_labels(labels, predicted_clusters)

    out_dir = args.out_root / args.section / args.fold
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "spot_id": spot_ids,
            "pixel_x": coords[:, 0],
            "pixel_y": coords[:, 1],
            "manual_label": labels,
            "measured_cluster": measured_clusters,
            "predicted_cluster": predicted_clusters,
        }
    ).to_csv(out_dir / f"{args.section}_loso_cluster_assignments.csv", index=False)
    (out_dir / f"{args.section}_loso_cluster_metrics.json").write_text(
        json.dumps(
            {
                "section": args.section,
                "fold": args.fold,
                "k": args.k,
                "top_n_variable_genes": int(len(top_idx)),
                "measured_expression": measured_eval,
                "predicted_expression": predicted_eval,
                "prediction_metrics": metrics,
                "note": "H1 was held out during model training and early stopping. Clustering used top measured-variable genes.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_summary(
        out_dir,
        args.section,
        he,
        coords,
        labels,
        measured_clusters,
        predicted_clusters,
        measured_eval,
        predicted_eval,
        corr_all,
        corr_hvg,
        metrics,
        args.k,
        args.dpi,
    )
    print(out_dir)
    print(json.dumps({"measured": measured_eval, "predicted": predicted_eval}, indent=2))


if __name__ == "__main__":
    main()
