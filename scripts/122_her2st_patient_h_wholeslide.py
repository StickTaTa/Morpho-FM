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
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
from skimage.transform import resize

import prostate_protocol_b_morpho as core


ROOT = Path(os.environ.get("MORPHO_VC_ROOT", Path(__file__).resolve().parents[1])).resolve()
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from st_pipeline.data.gene_vocab import load_gene_vocab, map_genes_to_vocab
from st_pipeline.superres.feature_cache import ensure_feature_cache, load_cache_slide, normalize_slide


HER2_SLIDES = ["H1", "H2", "H3"]
GENE_VOCAB_PATH = ROOT / "assets" / "cellfm" / "expand_gene_info.csv"
OUT_ROOT = ROOT / "results" / "her2st_patient_h" / "whole_slide"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HER2ST patient H whole-slide dense maps for selected genes.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--train-slide", choices=HER2_SLIDES, required=True)
    parser.add_argument("--section", choices=HER2_SLIDES, default=None, help="Slide to decode. Defaults to --train-slide.")
    parser.add_argument("--tag", default=None, help="Checkpoint tag. Defaults to train_{train-slide}_to_{section}.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Explicit checkpoint directory.")
    parser.add_argument("--out-root", type=Path, default=None, help="Override output root.")
    parser.add_argument("--genes", default=None, help="Comma-separated genes to decode.")
    parser.add_argument("--genes-file", type=Path, default=None, help="Plain-text gene list, one gene per line.")
    parser.add_argument("--top-n", type=int, default=24, help="Fallback number of high-variance genes if no gene list is supplied.")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-dim", type=int, default=16000)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def default_tag(train_slide: str, section: str) -> str:
    return f"train_{train_slide}_to_{section}"


def checkpoint_dir(root: Path, tag: str) -> Path:
    return root / "checkpoints" / "her2st_patient_h" / "st_superres" / "single_slide" / tag


def single_vrange(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(arr, [1, 99])
    vmin = max(0.0, float(vmin))
    vmax = float(vmax)
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(arr))
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
    return vmin, vmax


def upsample_map(map_small: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    valid = np.isfinite(map_small).astype(np.float32)
    values = np.nan_to_num(map_small, nan=0.0).astype(np.float32)
    up_values = resize(values, target_hw, order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)
    up_valid = resize(valid, target_hw, order=0, preserve_range=True, anti_aliasing=False) > 0.5
    up_values[~up_valid] = np.nan
    return up_values


def resolve_genes(root: Path, train_slide: str, section: str, ckpt_dir: Path, args: argparse.Namespace) -> list[str]:
    if args.genes_file is not None:
        return [g.strip() for g in args.genes_file.read_text(encoding="utf-8").splitlines() if g.strip()]
    if args.genes:
        return [g.strip() for g in args.genes.split(",") if g.strip()]

    common_genes = json.loads((ckpt_dir / "common_genes.json").read_text(encoding="utf-8"))
    adata = ad.read_h5ad(root / "data" / "spatial_data" / f"{section}.h5ad")
    present = [g for g in common_genes if g in adata.var_names]
    if not present:
        raise ValueError(f"No common genes from checkpoint found in {section}.h5ad")
    X = adata[:, present].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    vars_ = X.var(axis=0)
    order = np.argsort(-vars_)
    top_n = min(args.top_n, len(present))
    genes = [present[i] for i in order[:top_n]]
    print(f"[gene selection] top {top_n} variable genes from {section}: {genes[:8]}{' ...' if len(genes) > 8 else ''}")
    return genes


def resolve_checkpoint_dir(root: Path, tag: str, args: argparse.Namespace) -> Path:
    return args.checkpoint_dir.resolve() if args.checkpoint_dir is not None else checkpoint_dir(root, tag)


def dense_predict(root: Path, train_slide: str, section: str, tag: str, genes: list[str], args: argparse.Namespace, device: torch.device):
    ckpt_dir = resolve_checkpoint_dir(root, tag, args)
    out_dir = OUT_ROOT / tag / section
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "pred_superres_selected.npy"
    gene_path = out_dir / "pred_superres_selected_genes.json"
    meta_path = out_dir / "pred_superres_selected_meta.json"

    if not args.force and pred_path.exists() and gene_path.exists():
        saved_genes = json.loads(gene_path.read_text(encoding="utf-8"))
        if saved_genes == genes:
            pred_maps = np.load(pred_path, mmap_mode="r")
            return pred_maps, saved_genes, json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    emb_path = root / "data" / "superres_cache" / section / "embeddings-hist.pickle"
    if args.force_cache or not emb_path.exists():
        ensure_feature_cache([section], root=root, force=args.force_cache, device=device, max_dim=args.max_dim, use_tissue_seg=True)
    else:
        print(f"[{section}] using existing feature cache: {emb_path}")

    vocab = load_gene_vocab(GENE_VOCAB_PATH)
    kept_genes, gene_ids = map_genes_to_vocab(genes, vocab)
    if kept_genes != genes:
        missing = [g for g in genes if g not in kept_genes]
        raise ValueError(f"Genes missing from CellFM vocab: {missing}")

    common_genes = json.loads((ckpt_dir / "common_genes.json").read_text(encoding="utf-8"))
    embs, cnts, _, _, _, _ = load_cache_slide(section, root=root, genes=kept_genes)
    embs_norm, _, _ = normalize_slide(embs, cnts)

    cache_dir = root / "data" / "superres_cache" / section
    mask_small = np.array(Image.open(cache_dir / "mask-small.png")) > 0
    h, w, channels = embs_norm.shape
    if mask_small.shape != (h, w):
        mask_small = resize(mask_small.astype(np.float32), (h, w), order=0, preserve_range=True, anti_aliasing=False) > 0.5

    finite_mask = np.isfinite(embs_norm).all(-1)
    valid_mask = finite_mask & mask_small
    valid_idx = np.flatnonzero(valid_mask.ravel())
    valid_embs = embs_norm.reshape(-1, channels)[valid_idx].astype(np.float32)

    model = core.build_model(channels, len(common_genes), len(vocab), root, device)
    state = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    pred_valid = np.empty((len(valid_embs), len(genes)), dtype=np.float32)
    gene_ids_t = torch.as_tensor(gene_ids, device=device, dtype=torch.long)
    use_amp = device.type == "cuda"

    with torch.inference_mode():
        for start in range(0, len(valid_embs), args.batch_size):
            end = min(start + args.batch_size, len(valid_embs))
            x = torch.from_numpy(valid_embs[start:end]).to(device)
            gene_ids_batch = gene_ids_t.unsqueeze(0).expand(x.shape[0], -1)
            with torch.amp.autocast("cuda", enabled=use_amp):
                z = model.adapter(x)
                cell_emb, gene_emb = model.cellfm(z, gene_ids_batch)
                pred = model.cellfm.cellwise_decode(cell_emb, gene_emb)
                pred = torch.nn.functional.softplus(pred)
            pred_valid[start:end] = pred.float().cpu().numpy()
            if start == 0 or (start // args.batch_size) % 10 == 0:
                print(f"[{section}] dense decode {end}/{len(valid_embs)}")

    pred_maps = np.full((h, w, len(genes)), np.nan, dtype=np.float32)
    pred_maps.reshape(-1, len(genes))[valid_idx] = pred_valid

    np.save(pred_path, pred_maps)
    gene_path.write_text(json.dumps(genes, indent=2), encoding="utf-8")
    meta = {
        "train_slide": train_slide,
        "section": section,
        "tag": tag,
        "valid_grid_points": int(len(valid_idx)),
        "height": int(h),
        "width": int(w),
        "n_genes": int(len(genes)),
        "batch_size": int(args.batch_size),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[saved] {pred_path}")
    return pred_maps, genes, meta


def render_panels(root: Path, section: str, tag: str, genes: list[str], pred_maps: np.ndarray) -> Path:
    out_dir = OUT_ROOT / tag / section
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    he_img = np.array(Image.open(root / "data" / "superres_cache" / section / "he.jpg"))
    _, measured_cnts, locs_px, _, _, _ = load_cache_slide(section, root=root, genes=genes)
    manifest_rows = []
    for idx, gene in enumerate(genes):
        dense_map = np.asarray(pred_maps[..., idx], dtype=np.float32)
        dense_up = upsample_map(dense_map, he_img.shape[:2])
        measured = measured_cnts[:, idx]

        measured_vmin, measured_vmax = single_vrange(measured)
        dense_vmin, dense_vmax = single_vrange(dense_up)

        fig = Figure(figsize=(15.0, 5.2))
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, 3)
        titles = ["H&E", f"Measured sparse obs\n{gene}", f"Dense whole-slide\n{gene}"]
        for ax, title in zip(axes, titles):
            ax.imshow(he_img)
            ax.set_title(title, fontsize=11)
            ax.axis("off")

        x = locs_px[:, 1]
        y = locs_px[:, 0]
        sc = axes[1].scatter(
            x,
            y,
            c=measured,
            s=8,
            alpha=0.85,
            cmap="viridis",
            vmin=measured_vmin,
            vmax=measured_vmax,
            edgecolors="none",
            rasterized=True,
        )
        cbar1 = fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.03)
        cbar1.ax.tick_params(labelsize=8)

        alpha = np.where(np.isfinite(dense_up), 0.92, 0.0)
        im = axes[2].imshow(dense_up, cmap="viridis", vmin=dense_vmin, vmax=dense_vmax, alpha=alpha)
        cbar2 = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03)
        cbar2.ax.tick_params(labelsize=8)

        fig.suptitle(f"{section} | {gene}", fontsize=14, y=0.98)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.08)
        out_pdf = fig_dir / f"{gene}_dense_panel.pdf"
        out_png = fig_dir / f"{gene}_dense_panel.png"
        fig.savefig(out_pdf, dpi=220, bbox_inches="tight")
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        fig.clear()
        print(f"[saved] {out_pdf}")

        manifest_rows.append(
            {
                "section": section,
                "gene": gene,
                "panel_pdf": str(out_pdf),
                "panel_png": str(out_png),
                "measured_vmin": measured_vmin,
                "measured_vmax": measured_vmax,
                "dense_vmin": dense_vmin,
                "dense_vmax": dense_vmax,
            }
        )

    manifest_path = fig_dir / "dense_panel_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"[saved] {manifest_path}")
    return manifest_path


def main() -> None:
    global OUT_ROOT
    args = parse_args()
    root = args.root.resolve()
    if args.out_root is not None:
        OUT_ROOT = args.out_root.resolve()
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
    section = args.section or args.train_slide
    tag = args.tag or default_tag(args.train_slide, section)

    if args.predict_only and args.plot_only:
        raise ValueError("--predict-only and --plot-only cannot be used together.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("ROOT:", root)
    print("Device:", device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("GPU:", torch.cuda.get_device_name(0))
    elif not args.allow_cpu:
        raise RuntimeError("CUDA device not available. Re-run on GPU or pass --allow-cpu intentionally.")

    ckpt_dir = resolve_checkpoint_dir(root, tag, args)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    genes = resolve_genes(root, args.train_slide, section, ckpt_dir, args)
    if not args.plot_only:
        pred_maps, genes, meta = dense_predict(root, args.train_slide, section, tag, genes, args, device)
        (OUT_ROOT / tag / section / "run_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    else:
        pred_maps = np.load(OUT_ROOT / tag / section / "pred_superres_selected.npy", mmap_mode="r")
        genes = json.loads((OUT_ROOT / tag / section / "pred_superres_selected_genes.json").read_text(encoding="utf-8"))

    if not args.predict_only:
        render_panels(root, section, tag, genes, pred_maps)


if __name__ == "__main__":
    main()
