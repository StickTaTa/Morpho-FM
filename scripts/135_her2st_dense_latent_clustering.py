from __future__ import annotations

import argparse
import json
import sys
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
from scipy.ndimage import median_filter


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import prostate_protocol_b_morpho as core  # noqa: E402
from st_pipeline.data.gene_vocab import load_gene_vocab, map_genes_to_vocab  # noqa: E402
from st_pipeline.superres.feature_cache import ensure_feature_cache, load_cache_slide, normalize_slide  # noqa: E402


OUT_ROOT = ROOT / "results" / "her2st_patient_h" / "dense_latent_clustering"
SLIDES = ("H1", "H2", "H3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster HER2ST whole-slide CellFM latent embeddings on the dense tissue grid."
    )
    parser.add_argument("--slides", nargs="+", default=list(SLIDES), choices=SLIDES)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Explicit checkpoint directory for all requested slides.")
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Cluster count. Defaults to number of labeled classes on H1 or 7 otherwise.",
    )
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--sample-size", type=int, default=50000, help="Maximum latent vectors used for PCA/k-means fitting.")
    parser.add_argument("--kmeans-iters", type=int, default=35)
    parser.add_argument("--display-median-size", type=int, default=3, help="Median filter size for categorical display only.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def checkpoint_dir(slide: str) -> Path:
    return ROOT / "checkpoints" / "her2st_patient_h" / "st_superres" / "single_slide" / f"train_{slide}_to_{slide}"


def resolve_checkpoint_dir(slide: str, args: argparse.Namespace) -> Path:
    if args.checkpoint_dir is not None:
        return args.checkpoint_dir.resolve()
    return checkpoint_dir(slide)


def choose_n_clusters(slide: str, requested: int | None) -> int:
    if requested is not None:
        return int(requested)
    if slide == "H1":
        try:
            import anndata as ad

            adata = ad.read_h5ad(ROOT / "data" / "spatial_data" / "H1.h5ad")
            labels = adata.obs["label"].astype(str)
            labels = labels[~labels.isin(["undetermined", "unlabeled"])]
            n = int(labels.nunique())
            if n > 1:
                return n
        except Exception as exc:
            print(f"[H1] failed to infer labeled class count, fallback to 7: {exc}")
    return 7


def build_valid_mask(section: str, embs_norm: np.ndarray) -> np.ndarray:
    cache_dir = ROOT / "data" / "superres_cache" / section
    mask_small = np.array(Image.open(cache_dir / "mask-small.png")) > 0
    if mask_small.shape != embs_norm.shape[:2]:
        mask_img = Image.fromarray((mask_small.astype(np.uint8) * 255))
        mask_img = mask_img.resize((embs_norm.shape[1], embs_norm.shape[0]), resample=Image.Resampling.NEAREST)
        mask_small = np.array(mask_img) > 0
    finite = np.isfinite(embs_norm).all(-1)
    return finite & mask_small


def extract_dense_cellfm_latent(
    model,
    valid_embs: np.ndarray,
    gene_id: int,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
) -> np.ndarray:
    latent_chunks: list[np.ndarray] = []
    gene_id_t = torch.as_tensor([gene_id], device=device, dtype=torch.long)
    with torch.inference_mode():
        for start in range(0, len(valid_embs), batch_size):
            end = min(start + batch_size, len(valid_embs))
            x = torch.from_numpy(valid_embs[start:end].astype(np.float32)).to(device)
            gene_ids_batch = gene_id_t.unsqueeze(0).expand(x.shape[0], -1)
            with torch.amp.autocast("cuda", enabled=use_amp):
                z = model.adapter(x)
                cell_emb, _ = model.cellfm(z, gene_ids_batch)
            latent_chunks.append(cell_emb.detach().float().cpu().numpy())
            if end == len(valid_embs) or start == 0 or (end // batch_size) % 10 == 0:
                print(f"    latent {end:,}/{len(valid_embs):,}")
    return np.concatenate(latent_chunks, axis=0).astype(np.float32, copy=False)


def cluster_latents_torch(
    latents: np.ndarray,
    n_clusters: int,
    sample_size: int,
    n_iters: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(42)
    n = len(latents)
    fit_idx = np.arange(n)
    if sample_size > 0 and n > sample_size:
        fit_idx = rng.choice(n, size=sample_size, replace=False)

    mean = latents.mean(axis=0, keepdims=True).astype(np.float32)
    std = (latents.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    fit = (latents[fit_idx].astype(np.float32) - mean) / std
    fit_t = torch.from_numpy(fit).to(device)
    # Cosine-like clustering after L2 normalization is more stable for CellFM embeddings.
    fit_t = torch.nn.functional.normalize(fit_t, dim=1)
    perm = torch.randperm(fit_t.shape[0], device=device)
    centers = fit_t[perm[:n_clusters]].clone()

    for _ in range(n_iters):
        labels_fit = torch.empty(fit_t.shape[0], dtype=torch.long, device=device)
        for start in range(0, fit_t.shape[0], 16384):
            end = min(start + 16384, fit_t.shape[0])
            # For normalized vectors, maximizing dot product is equivalent to cosine nearest centroid.
            labels_fit[start:end] = torch.argmax(fit_t[start:end] @ centers.T, dim=1)
        new_centers = []
        for k in range(n_clusters):
            mask = labels_fit == k
            if bool(mask.any()):
                new_centers.append(fit_t[mask].mean(dim=0))
            else:
                new_centers.append(fit_t[torch.randint(0, fit_t.shape[0], (1,), device=device)].squeeze(0))
        centers = torch.stack(new_centers, dim=0)
        centers = torch.nn.functional.normalize(centers, dim=1)

    labels = np.empty(n, dtype=np.int16)
    mean_t = torch.from_numpy(mean).to(device)
    std_t = torch.from_numpy(std).to(device)
    with torch.inference_mode():
        for start in range(0, n, 32768):
            end = min(start + 32768, n)
            x = torch.from_numpy(latents[start:end].astype(np.float32)).to(device)
            x = torch.nn.functional.normalize((x - mean_t) / std_t, dim=1)
            labels[start:end] = torch.argmax(x @ centers.T, dim=1).detach().cpu().numpy().astype(np.int16)

    stats = {
        "fit_sample_size": int(len(fit_idx)),
        "latent_dim": int(latents.shape[1]),
        "kmeans_iters": int(n_iters),
        "method": "torch_cosine_kmeans_on_standardized_cellfm_latents",
    }
    return labels, stats


def display_label_map(raw_map: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return raw_map.copy()
    encoded = np.where(np.isfinite(raw_map), raw_map.astype(np.int16), -1)
    filtered = median_filter(encoded, size=size, mode="nearest")
    out = np.where(encoded >= 0, filtered, -1).astype(np.float32)
    out[out < 0] = np.nan
    return out


def label_rgba(label_map: np.ndarray, out_hw: tuple[int, int], n_clusters: int) -> np.ndarray:
    encoded = np.where(np.isfinite(label_map), label_map.astype(np.int16), -1)
    up_img = Image.fromarray(encoded)
    up_img = up_img.resize((out_hw[1], out_hw[0]), resample=Image.Resampling.NEAREST)
    up = np.array(up_img)
    cmap = plt.get_cmap("tab10", n_clusters)
    rgba = np.zeros((*out_hw, 4), dtype=np.float32)
    valid = up >= 0
    rgba[valid] = cmap(up[valid].astype(int) % n_clusters)
    rgba[valid, 3] = 0.70
    return rgba


def save_single_overlay(
    slide: str,
    he_img: np.ndarray,
    label_map: np.ndarray,
    n_clusters: int,
    out_dir: Path,
    dpi: int,
) -> None:
    rgba = label_rgba(label_map, he_img.shape[:2], n_clusters)
    cmap = plt.get_cmap("tab10", n_clusters)
    fig, ax = plt.subplots(figsize=(6.7, 4.8))
    ax.imshow(he_img)
    ax.imshow(rgba)
    ax.set_title(f"{slide} dense latent domains", fontsize=10)
    ax.axis("off")
    handles = [mpatches.Patch(color=cmap(i), label=f"Domain {i}") for i in range(n_clusters)]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        frameon=False,
    )
    fig.savefig(out_dir / f"{slide}_dense_latent_domains.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{slide}_dense_latent_domains.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_combined_overlay(
    overlays: dict[str, tuple[np.ndarray, np.ndarray, int]],
    out_dir: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, len(overlays), figsize=(6.5, 2.35))
    if len(overlays) == 1:
        axes = [axes]
    cluster_counts = []
    for ax, (slide, (he_img, label_map, n_clusters)) in zip(axes, overlays.items()):
        cluster_counts.append(n_clusters)
        ax.imshow(he_img)
        ax.imshow(label_rgba(label_map, he_img.shape[:2], n_clusters))
        ax.set_title(f"{slide} (k={n_clusters})", fontsize=10)
        ax.axis("off")
    if len(set(cluster_counts)) == 1:
        n_clusters = cluster_counts[0]
        cmap = plt.get_cmap("tab10", n_clusters)
        handles = [mpatches.Patch(color=cmap(i), label=f"Domain {i}") for i in range(n_clusters)]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(n_clusters, 7),
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )
    fig.suptitle("HER2ST dense latent candidate domains", fontsize=11, y=0.98)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.18, wspace=0.03)
    fig.savefig(out_dir / "HER2ST_dense_latent_candidate_domains.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "HER2ST_dense_latent_candidate_domains.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / "HER2ST_dense_latent_candidate_domains.tif", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_slide(slide: str, args: argparse.Namespace, device: torch.device, use_amp: bool) -> tuple[np.ndarray, np.ndarray, int]:
    ckpt_dir = resolve_checkpoint_dir(slide, args)
    out_dir = OUT_ROOT / slide
    out_dir.mkdir(parents=True, exist_ok=True)
    n_clusters = choose_n_clusters(slide, args.n_clusters)
    raw_map_path = out_dir / f"{slide}_dense_latent_domain_map_raw.npy"
    display_map_path = out_dir / f"{slide}_dense_latent_domain_map_display.npy"
    if raw_map_path.exists() and display_map_path.exists() and not args.force:
        print(f"[{slide}] using existing clustering maps")
        raw_map = np.load(raw_map_path)
        display_map = np.load(display_map_path)
        meta_path = out_dir / f"{slide}_dense_latent_cluster_meta.json"
        if meta_path.exists():
            try:
                n_clusters = int(json.loads(meta_path.read_text(encoding="utf-8")).get("n_clusters", n_clusters))
            except Exception:
                pass
        return raw_map, display_map, n_clusters

    print(f"[{slide}] checkpoint={ckpt_dir} n_clusters={n_clusters}")
    emb_path = ROOT / "data" / "superres_cache" / slide / "embeddings-hist.pickle"
    if emb_path.exists():
        print(f"[{slide}] using existing feature cache: {emb_path}")
    else:
        ensure_feature_cache(
            [slide],
            root=ROOT,
            force=False,
            device=str(device),
            max_dim=16000,
            use_tissue_seg=True,
        )
    common_genes = json.loads((ckpt_dir / "common_genes.json").read_text(encoding="utf-8"))
    vocab = load_gene_vocab(ROOT / "assets" / "cellfm" / "expand_gene_info.csv")
    kept_genes, gene_ids = map_genes_to_vocab(common_genes, vocab)
    embs, cnts, _, _, _, _ = load_cache_slide(slide, root=ROOT, genes=kept_genes)
    embs_norm, _, _ = normalize_slide(embs, cnts)

    valid_mask = build_valid_mask(slide, embs_norm)
    valid_idx = np.flatnonzero(valid_mask.ravel())
    channels = embs_norm.shape[-1]
    valid_embs = embs_norm.reshape(-1, channels)[valid_idx].astype(np.float32)
    print(f"[{slide}] valid grid points={len(valid_idx):,}, channels={channels}")

    model = core.build_model(channels, len(common_genes), len(vocab), ROOT, device)
    state = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    latents = extract_dense_cellfm_latent(
        model=model,
        valid_embs=valid_embs,
        gene_id=int(gene_ids[0]),
        device=device,
        batch_size=args.batch_size,
        use_amp=use_amp,
    )
    print(f"[{slide}] latent shape={latents.shape}")
    labels, cluster_stats = cluster_latents_torch(
        latents,
        n_clusters=n_clusters,
        sample_size=args.sample_size,
        n_iters=args.kmeans_iters,
        device=device,
    )

    raw_map = np.full(embs_norm.shape[:2], np.nan, dtype=np.float32)
    raw_map.reshape(-1)[valid_idx] = labels.astype(np.float32)
    display_map = display_label_map(raw_map, size=args.display_median_size)
    np.save(raw_map_path, raw_map)
    np.save(display_map_path, display_map)
    pd.DataFrame(
        {
            "grid_index": valid_idx.astype(np.int64),
            "grid_y": (valid_idx // embs_norm.shape[1]).astype(np.int32),
            "grid_x": (valid_idx % embs_norm.shape[1]).astype(np.int32),
            "domain": labels.astype(np.int16),
        }
    ).to_csv(out_dir / f"{slide}_dense_latent_domain_assignments.csv", index=False)
    (out_dir / f"{slide}_dense_latent_cluster_meta.json").write_text(
        json.dumps(
            {
                "slide": slide,
                "n_clusters": n_clusters,
                "sample_size": args.sample_size,
                "kmeans_iters": args.kmeans_iters,
                "display_median_size": args.display_median_size,
                "valid_grid_points": int(len(valid_idx)),
                "checkpoint_dir": str(ckpt_dir),
                **cluster_stats,
                "note": (
                    "Domain labels were fitted on dense CellFM latent embeddings. "
                    "The display map applies a median filter only for categorical visualization; "
                    "raw assignments are saved separately."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return raw_map, display_map, n_clusters


def main() -> None:
    global ROOT, OUT_ROOT
    args = parse_args()
    ROOT = args.root.resolve()
    OUT_ROOT = args.out_root.resolve()
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA is not available. Pass --allow-cpu only if a slow CPU run is intended.")
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"[device] {torch.cuda.get_device_name(0)}")
    else:
        print("[device] CPU")

    combined: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    rows: list[dict[str, object]] = []
    for slide in args.slides:
        raw_map, display_map, n_clusters = run_slide(slide, args=args, device=device, use_amp=use_amp)
        he_img = np.array(Image.open(ROOT / "data" / "superres_cache" / slide / "he.jpg"))
        save_single_overlay(slide, he_img, display_map, n_clusters, OUT_ROOT / slide, args.dpi)
        combined[slide] = (he_img, display_map, n_clusters)
        rows.append(
            {
                "slide": slide,
                "n_clusters": int(n_clusters),
                "n_raw_assigned_grid_points": int(np.isfinite(raw_map).sum()),
                "n_display_assigned_grid_points": int(np.isfinite(display_map).sum()),
                "single_pdf": str((OUT_ROOT / slide / f"{slide}_dense_latent_domains.pdf")),
                "single_png": str((OUT_ROOT / slide / f"{slide}_dense_latent_domains.png")),
            }
        )

    save_combined_overlay(combined, OUT_ROOT, args.dpi)
    pd.DataFrame(rows).to_csv(OUT_ROOT / "dense_latent_clustering_manifest.csv", index=False)
    print(f"[done] {OUT_ROOT}")


if __name__ == "__main__":
    main()
