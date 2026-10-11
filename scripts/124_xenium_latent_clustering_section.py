from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset


ROOT = Path(os.environ.get("MORPHO_VC_ROOT", r"E:\Morpho-FM")).resolve()
sys.path.append(str(ROOT / "src"))

from st_pipeline.constants import KEYS  # noqa: E402
from st_pipeline.data.collate import mil_collate  # noqa: E402
from st_pipeline.data.gene_vocab import load_gene_vocab, map_genes_to_vocab  # noqa: E402
from st_pipeline.model.morpho_cellfm_mil import MorphoCellfmMIL  # noqa: E402
from st_pipeline.superres.feature_cache import (  # noqa: E402
    ensure_feature_cache,
    load_cache_slide,
    normalize_slide,
)


GRID_FACTOR = 16


def get_disk_mask(radius: float) -> np.ndarray:
    radius_ceil = int(np.ceil(radius))
    locs = np.meshgrid(
        np.arange(-radius_ceil, radius_ceil + 1),
        np.arange(-radius_ceil, radius_ceil + 1),
        indexing="ij",
    )
    locs = np.stack(locs, axis=-1)
    return (locs**2).sum(-1) <= radius**2


class SuperresSpotDataset(Dataset):
    def __init__(
        self,
        embs: np.ndarray,
        cnts: np.ndarray,
        locs_px: np.ndarray,
        radius_px: float,
        gene_ids: np.ndarray,
        grid_factor: int = GRID_FACTOR,
    ) -> None:
        super().__init__()
        self.embs = embs
        self.cnts = cnts
        self.gene_ids = gene_ids
        self.locs_px = np.asarray(locs_px, dtype=np.float32).copy()
        locs_grid = np.round(self.locs_px / float(grid_factor)).astype(np.int64)

        self.radius = float(radius_px) / float(grid_factor)
        self.mask = get_disk_mask(self.radius)

        h, w = embs.shape[:2]
        shape = np.array(self.mask.shape)
        center = shape // 2
        r0 = -center
        r1 = shape - center

        valid = (
            (locs_grid[:, 0] + r0[0] >= 0)
            & (locs_grid[:, 0] + r1[0] <= h)
            & (locs_grid[:, 1] + r0[1] >= 0)
            & (locs_grid[:, 1] + r1[1] <= w)
        )
        finite_center = np.isfinite(embs[locs_grid[:, 0], locs_grid[:, 1]]).all(-1)
        valid &= finite_center

        self.keep_idx = np.where(valid)[0]
        self.locs = locs_grid[valid]
        self.locs_px = self.locs_px[valid]
        self.cnts = cnts[valid]

        sums = self.cnts.sum(axis=1)
        self.size_factors = (sums / (sums.mean() + 1e-8)).astype(np.float32)

    def __len__(self) -> int:
        return len(self.locs)

    def __getitem__(self, idx: int) -> dict:
        i, j = self.locs[idx]
        shape = np.array(self.mask.shape)
        center = shape // 2
        r0 = -center
        r1 = shape - center
        patch = self.embs[i + r0[0] : i + r1[0], j + r0[1] : j + r1[1]]
        x = patch if self.mask.all() else patch[self.mask]
        return {
            KEYS.X: torch.from_numpy(x.astype(np.float32)),
            KEYS.Y_BAG: torch.from_numpy(self.cnts[idx].astype(np.float32)),
            KEYS.SIZE_FACTOR: float(self.size_factors[idx]),
            KEYS.GENE_IDS: self.gene_ids,
        }


def load_model(
    input_dim: int,
    n_genes: int,
    vocab_size: int,
    ckpt_dir: Path,
    cellfm_checkpoint: Path,
    device: torch.device,
) -> MorphoCellfmMIL:
    model = MorphoCellfmMIL(
        input_dim=input_dim,
        n_genes=n_genes,
        cellfm_dim=1536,
        cellfm_layers=2,
        cellfm_heads=48,
        cellfm_checkpoint=str(cellfm_checkpoint),
        freeze_cellfm=True,
        use_mock=False,
        use_retention=True,
        vocab_size=vocab_size,
        dropout=0.1,
        aggregation="mean",
        dispersion="gene",
    ).to(device)
    ckpt_path = ckpt_dir / "best_model.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def extract_spot_latents(
    model: MorphoCellfmMIL,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    chunks = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, start=1):
            x = batch[KEYS.X].to(device, non_blocking=True)
            gene_ids = batch[KEYS.GENE_IDS].to(device, non_blocking=True)
            if gene_ids.dim() == 1:
                gene_ids = gene_ids.unsqueeze(0).expand(x.size(0), -1)
            gene_ids_dummy = gene_ids[:, :1]
            with torch.amp.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                enabled=use_amp,
            ):
                prompt = model.adapter(x)
                cell_emb, _ = model.cellfm(prompt, gene_ids_dummy)
            chunks.append(cell_emb.detach().float().cpu().numpy())
            if batch_idx % 25 == 0:
                n_seen = sum(chunk.shape[0] for chunk in chunks)
                print(f"[spot latent] {n_seen:,} cells in {time.perf_counter() - t0:.1f}s")
    return np.concatenate(chunks, axis=0)


def predict_full_slide_clusters(
    model: MorphoCellfmMIL,
    embs_norm: np.ndarray,
    gene_ids: np.ndarray,
    pca: PCA,
    kmeans: MiniBatchKMeans,
    valid_indices: np.ndarray,
    device: torch.device,
    use_amp: bool,
    chunk_size: int,
) -> np.ndarray:
    labels = np.empty(len(valid_indices), dtype=np.int16)
    gene_id_dummy = torch.tensor([[int(gene_ids[0])]], device=device, dtype=torch.long)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(valid_indices), chunk_size):
            end = min(start + chunk_size, len(valid_indices))
            idx = valid_indices[start:end]
            emb_chunk = embs_norm[idx[:, 0], idx[:, 1]]
            x = torch.from_numpy(emb_chunk.astype(np.float32)).unsqueeze(1).to(device)
            gene_ids_chunk = gene_id_dummy.expand(x.shape[0], -1)
            with torch.amp.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                enabled=use_amp,
            ):
                prompt = model.adapter(x)
                cell_emb, _ = model.cellfm(prompt, gene_ids_chunk)
            latent = cell_emb.detach().float().cpu().numpy()
            latent_pca = pca.transform(latent)
            labels[start:end] = kmeans.predict(latent_pca).astype(np.int16)
            if end == len(valid_indices) or (end // chunk_size) % 20 == 0:
                print(f"[full-slide latent] {end:,}/{len(valid_indices):,} in {time.perf_counter() - t0:.1f}s")
    return labels


def build_valid_grid_mask(cache_dir: Path, embs_norm: np.ndarray) -> np.ndarray:
    tissue_mask = None
    mask_path = cache_dir / "mask-small.png"
    if mask_path.exists():
        tissue_mask = np.array(Image.open(mask_path)) > 0
        if tissue_mask.shape != embs_norm.shape[:2]:
            tissue_mask = resize(
                tissue_mask.astype(float),
                embs_norm.shape[:2],
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            ) > 0.5
    finite_mask = np.isfinite(embs_norm).all(-1)
    return finite_mask & tissue_mask if tissue_mask is not None else finite_mask


def plot_spot_overlay(
    he_img: np.ndarray,
    locs_px: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    section: str,
    out_dir: Path,
) -> None:
    cmap = plt.get_cmap("tab20", n_clusters)
    fig, ax = plt.subplots(figsize=(24, 18))
    ax.imshow(he_img)
    ax.scatter(
        locs_px[:, 1],
        locs_px[:, 0],
        c=labels,
        s=8,
        alpha=0.9,
        cmap="tab20",
        edgecolors="none",
    )
    handles = [mpatches.Patch(color=cmap(i), label=f"Domain {i}") for i in range(n_clusters)]
    ax.legend(
        handles=handles,
        title="Latent domains",
        bbox_to_anchor=(1.05, 1.0),
        loc="upper left",
        prop={"size": 18},
        title_fontsize=20,
        frameon=False,
    )
    ax.set_title(f"{section} - Morphological Latent Domain Clustering (n={n_clusters})", fontsize=28, pad=20)
    ax.axis("off")
    fig.savefig(out_dir / "latent_clustering_overlay.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "latent_clustering_overlay.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_cluster_rgba(
    cluster_map: np.ndarray,
    he_shape: tuple[int, int],
    n_clusters: int,
) -> np.ndarray:
    cluster_map_up = resize(
        cluster_map,
        he_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    cmap = plt.get_cmap("tab20", n_clusters)
    rgba = np.zeros((*he_shape, 4), dtype=np.float32)
    valid = ~np.isnan(cluster_map_up)
    rgba[valid] = cmap(cluster_map_up[valid].astype(int) % n_clusters)
    rgba[valid, 3] = 1.0
    return rgba


def plot_full_slide_overlay(
    he_img: np.ndarray,
    cluster_map: np.ndarray,
    n_clusters: int,
    section: str,
    out_dir: Path,
) -> None:
    cmap = plt.get_cmap("tab20", n_clusters)
    rgba = make_cluster_rgba(cluster_map, he_img.shape[:2], n_clusters)

    fig, ax = plt.subplots(figsize=(24, 18))
    ax.imshow(he_img)
    ax.imshow(rgba, alpha=0.7)
    handles = [mpatches.Patch(color=cmap(i), label=f"Domain {i}") for i in range(n_clusters)]
    ax.legend(
        handles=handles,
        title="Latent domains",
        bbox_to_anchor=(1.05, 1.0),
        loc="upper left",
        prop={"size": 18},
        title_fontsize=20,
        frameon=False,
    )
    ax.set_title(f"{section} - Full-slide Morphological Domain Clustering (n={n_clusters})", fontsize=28, pad=20)
    ax.axis("off")
    fig.savefig(out_dir / "full_slide_latent_clustering_overlay.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "full_slide_latent_clustering_overlay.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_overlay(
    he_img: np.ndarray,
    locs_px: np.ndarray,
    spot_labels: np.ndarray,
    cluster_map: np.ndarray,
    n_clusters: int,
    section: str,
    out_dir: Path,
) -> None:
    cmap = plt.get_cmap("tab20", n_clusters)
    rgba = make_cluster_rgba(cluster_map, he_img.shape[:2], n_clusters)

    fig, axes = plt.subplots(1, 2, figsize=(36, 14))
    axes[0].imshow(he_img)
    axes[0].scatter(
        locs_px[:, 1],
        locs_px[:, 0],
        c=spot_labels,
        s=4,
        alpha=0.9,
        cmap="tab20",
        edgecolors="none",
    )
    axes[0].set_title(f"Spot-level Clustering ({len(locs_px):,} Xenium spots)", fontsize=22)
    axes[0].axis("off")

    axes[1].imshow(he_img)
    axes[1].imshow(rgba, alpha=0.7)
    axes[1].set_title(f"Full-slide Grid Clustering ({np.isfinite(cluster_map).sum():,} tissue pixels)", fontsize=22)
    axes[1].axis("off")

    handles = [mpatches.Patch(color=cmap(i), label=f"Domain {i}") for i in range(n_clusters)]
    fig.legend(
        handles=handles,
        title="Latent domains",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        prop={"size": 16},
        title_fontsize=18,
        frameon=False,
    )
    fig.suptitle(
        f"{section} - Spot vs Full-slide Morphological Domain Clustering (n_clusters={n_clusters})",
        fontsize=26,
        y=1.02,
    )
    fig.savefig(out_dir / "grid_latent_clustering_overlay.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "grid_latent_clustering_overlay.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", default="XEN2", choices=["XEN1", "XEN2"])
    parser.add_argument("--n-clusters", type=int, default=15)
    parser.add_argument("--n-components", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--grid-chunk-size", type=int, default=4096)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=None,
        help=(
            "Model checkpoint directory. Defaults to the within-section checkpoint: "
            "xenium_cellfm for XEN1 and xenium_cellfm_2 for XEN2."
        ),
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = (not args.no_amp) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.ckpt_dir is not None:
        ckpt_dir = args.ckpt_dir.resolve()
    elif args.section == "XEN1":
        ckpt_dir = ROOT / "checkpoints" / "st_superres" / "xenium_cellfm"
    else:
        ckpt_dir = ROOT / "checkpoints" / "st_superres" / "xenium_cellfm_2"
    out_dir = ROOT / "results" / "st_superres" / "xenium_latent_clustering" / f"xenium_{args.section}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[config] ROOT={ROOT}")
    print(f"[config] section={args.section} device={device} amp={use_amp}")
    print(f"[config] ckpt_dir={ckpt_dir}")
    print(f"[config] output={out_dir}")

    t0 = time.perf_counter()
    ensure_feature_cache(
        [args.section],
        root=ROOT,
        force=args.force_cache,
        device=device,
        max_dim=16000,
        use_tissue_seg=True,
    )

    common_genes = json.loads((ckpt_dir / "common_genes.json").read_text())
    vocab = load_gene_vocab(ROOT / "assets" / "cellfm" / "expand_gene_info.csv")
    kept_genes, gene_ids = map_genes_to_vocab(common_genes, vocab)
    embs, cnts, locs, radius, gene_names, spot_ids = load_cache_slide(
        args.section,
        root=ROOT,
        genes=kept_genes,
    )
    embs_norm, _, stats = normalize_slide(embs, cnts)
    print(f"[cache] embs={embs.shape} cnts={cnts.shape} locs={locs.shape} radius={radius}")

    dataset = SuperresSpotDataset(embs_norm, cnts, locs, radius, gene_ids)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=mil_collate,
        pin_memory=(device.type == "cuda"),
    )
    sample = next(iter(loader))
    input_dim = sample[KEYS.X].shape[-1]
    model = load_model(
        input_dim=input_dim,
        n_genes=len(kept_genes),
        vocab_size=len(vocab),
        ckpt_dir=ckpt_dir,
        cellfm_checkpoint=ROOT / "checkpoints" / "CellFM" / "CellFM_80M_weight.pt",
        device=device,
    )

    spot_latents = extract_spot_latents(model, loader, device=device, use_amp=use_amp)
    np.save(out_dir / "spot_latent_features.npy", spot_latents.astype(np.float32))
    print(f"[spot latent] shape={spot_latents.shape}")

    pca = PCA(n_components=args.n_components, random_state=42)
    spot_pca = pca.fit_transform(spot_latents)
    np.save(out_dir / "spot_latent_pca.npy", spot_pca.astype(np.float32))
    pd.DataFrame(
        {
            "component": np.arange(1, args.n_components + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    ).to_csv(out_dir / "spot_latent_pca_variance.csv", index=False)

    kmeans = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        random_state=42,
        batch_size=2048,
        n_init="auto",
    )
    spot_labels = kmeans.fit_predict(spot_pca).astype(np.int16)
    np.save(out_dir / "latent_cluster_labels.npy", spot_labels)

    cache_dir = ROOT / "data" / "superres_cache" / args.section
    valid_mask = build_valid_grid_mask(cache_dir, embs_norm)
    valid_indices = np.argwhere(valid_mask)
    grid_labels = predict_full_slide_clusters(
        model,
        embs_norm=embs_norm,
        gene_ids=gene_ids,
        pca=pca,
        kmeans=kmeans,
        valid_indices=valid_indices,
        device=device,
        use_amp=use_amp,
        chunk_size=args.grid_chunk_size,
    )

    cluster_map = np.full(embs_norm.shape[:2], np.nan, dtype=np.float32)
    cluster_map[valid_indices[:, 0], valid_indices[:, 1]] = grid_labels.astype(np.float32)
    np.save(out_dir / "grid_latent_cluster_map.npy", cluster_map)
    np.save(out_dir / "grid_valid_indices.npy", valid_indices.astype(np.int32))
    np.save(out_dir / "grid_latent_cluster_labels.npy", grid_labels)

    Image.MAX_IMAGE_PIXELS = None
    he_img = np.array(Image.open(cache_dir / "he.jpg"))
    plot_spot_overlay(
        he_img,
        dataset.locs_px,
        spot_labels,
        args.n_clusters,
        args.section,
        out_dir,
    )
    plot_full_slide_overlay(he_img, cluster_map, args.n_clusters, args.section, out_dir)
    plot_comparison_overlay(
        he_img,
        dataset.locs_px,
        spot_labels,
        cluster_map,
        args.n_clusters,
        args.section,
        out_dir,
    )

    summary = pd.DataFrame(
        [
            {
                "section": args.section,
                "n_clusters": args.n_clusters,
                "n_components": args.n_components,
                "n_cache_cells": int(cnts.shape[0]),
                "n_valid_spot_cells": int(len(dataset)),
                "n_full_slide_grid_points": int(len(valid_indices)),
                "checkpoint_dir": str(ckpt_dir),
                "split_json": (ckpt_dir / "split.json").read_text(encoding="utf-8")
                if (ckpt_dir / "split.json").exists()
                else "",
                "elapsed_seconds": round(time.perf_counter() - t0, 2),
            }
        ]
    )
    summary.to_csv(out_dir / "latent_clustering_run_summary.csv", index=False)
    print("[done]", out_dir)


if __name__ == "__main__":
    main()
