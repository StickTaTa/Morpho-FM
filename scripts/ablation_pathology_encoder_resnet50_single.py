from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROSTATE_SLIDES = ["INT25", "INT26", "INT27", "INT28"]
GRID_FACTOR = 16
ENCODER_NAME = "resnet50_pca576"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pathology encoder ablation for prostate single-slide INT25->INT28: "
            "replace HIPT features with ImageNet ResNet50 features while keeping "
            "the Morpho-FM/CellFM backend fixed."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--mode", choices=["cache", "train", "predict", "summary", "all"], default="all")
    parser.add_argument("--train-slide", default="INT25", choices=PROSTATE_SLIDES)
    parser.add_argument("--test-slides", default="INT28")
    parser.add_argument("--tag", default=f"{ENCODER_NAME}_single_INT25_to_INT28")
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-predict", action="store_true")
    parser.add_argument("--keep-raw-resnet", action="store_true")
    parser.add_argument("--resnet-weights", type=Path, default=None)
    parser.add_argument("--pca-components", type=int, default=576)
    parser.add_argument("--pca-samples", type=int, default=100_000)
    parser.add_argument("--feature-tile-size", type=int, default=2048)
    parser.add_argument("--feature-batch-rows", type=int, default=16_384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--predict-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gene-chunk-size", type=int, default=1024)
    parser.add_argument("--gene-lr-mult", type=float, default=3.0)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-dim", type=int, default=16000)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--heartbeat-batches", type=int, default=5)
    return parser.parse_args()


def setup_root(root_arg: Path | None) -> Path:
    root = root_arg or Path(os.environ.get("MORPHO_VC_ROOT", "."))
    root = root.expanduser().resolve()
    os.chdir(root)
    for rel in ("src", "scripts"):
        p = root / rel
        if str(p) not in sys.path:
            sys.path.append(str(p))
    return root


def split_test_slides(arg: str) -> list[str]:
    slides = [x.strip() for x in arg.split(",") if x.strip()]
    unknown = sorted(set(slides) - set(PROSTATE_SLIDES))
    if unknown:
        raise ValueError(f"Unknown prostate slide IDs: {unknown}")
    return slides


def encoder_cache_root(root: Path, arg: Path | None) -> Path:
    return (arg or (root / "data" / "superres_cache_encoder_ablation" / ENCODER_NAME)).resolve()


def checkpoint_dir(root: Path, tag: str) -> Path:
    return root / "checkpoints" / "st_superres" / "pathology_encoder_ablation" / tag


def result_dir(root: Path, tag: str) -> Path:
    return root / "results" / "ablation_pathology_encoder" / tag


def load_base_image(path: Path) -> np.ndarray:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def pad_to_multiple(img: np.ndarray, multiple: int, pad_value: int = 255) -> np.ndarray:
    h, w = img.shape[:2]
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    if new_h == h and new_w == w:
        return img
    return np.pad(img, ((0, new_h - h), (0, new_w - w), (0, 0)), mode="constant", constant_values=pad_value)


def downsample_rgb(img: np.ndarray, factor: int = GRID_FACTOR) -> np.ndarray:
    img = pad_to_multiple(img, factor, pad_value=255)
    h, w = img.shape[:2]
    img = img.reshape(h // factor, factor, w // factor, factor, 3).astype(np.float32) / 255.0
    return img.mean(axis=(1, 3)).transpose(2, 0, 1).astype(np.float32)


def copy_slide_metadata(base_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "cache_meta.json",
        "cnts.tsv",
        "gene-names.txt",
        "he.jpg",
        "he-raw.jpg",
        "he-scaled.jpg",
        "locs.tsv",
        "mask-small.png",
        "radius.txt",
    ]:
        src = base_dir / name
        if src.exists():
            dst = out_dir / name
            if not dst.exists() or src.stat().st_size != dst.stat().st_size:
                shutil.copy2(src, dst)


def load_resnet_feature_model(weights_path: Path, device):
    import torch
    import torchvision.models as models

    model = models.resnet50(weights=None)
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    feature_model = torch.nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
    ).to(device)
    feature_model.eval()
    return feature_model


def extract_resnet_raw_features(
    slide_id: str,
    base_cache_root: Path,
    out_cache_root: Path,
    weights_path: Path,
    device: str,
    tile_size: int,
    force: bool,
) -> Path:
    import torch

    base_dir = base_cache_root / slide_id
    out_dir = out_cache_root / slide_id
    copy_slide_metadata(base_dir, out_dir)
    raw_path = out_dir / "resnet50_layer3_raw.npy"
    if raw_path.exists() and not force:
        print(f"[cache:{slide_id}] raw ResNet features already exist: {raw_path}")
        return raw_path

    img = load_base_image(base_dir / "he.jpg")
    grid_h = (img.shape[0] + GRID_FACTOR - 1) // GRID_FACTOR
    grid_w = (img.shape[1] + GRID_FACTOR - 1) // GRID_FACTOR
    img_pad = pad_to_multiple(img, tile_size, pad_value=255)
    raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float32, shape=(grid_h, grid_w, 1024))

    model = load_resnet_feature_model(weights_path, torch.device(device))
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    n_tiles_h = img_pad.shape[0] // tile_size
    n_tiles_w = img_pad.shape[1] // tile_size
    print(f"[cache:{slide_id}] extracting ResNet layer3 features from {n_tiles_h * n_tiles_w} tiles")
    with torch.inference_mode():
        for ti in range(n_tiles_h):
            for tj in range(n_tiles_w):
                y0 = ti * tile_size
                x0 = tj * tile_size
                tile = img_pad[y0 : y0 + tile_size, x0 : x0 + tile_size]
                x = torch.from_numpy(tile.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
                x = (x - mean) / std
                with torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
                    feat = model(x)
                feat_np = feat.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
                gy0 = y0 // GRID_FACTOR
                gx0 = x0 // GRID_FACTOR
                gy1 = min(gy0 + feat_np.shape[0], grid_h)
                gx1 = min(gx0 + feat_np.shape[1], grid_w)
                raw[gy0:gy1, gx0:gx1, :] = feat_np[: gy1 - gy0, : gx1 - gx0, :]
                if (ti * n_tiles_w + tj) % 10 == 0:
                    print(f"[cache:{slide_id}] tile {ti * n_tiles_w + tj + 1}/{n_tiles_h * n_tiles_w}", flush=True)
    raw.flush()
    print(f"[cache:{slide_id}] saved raw features: {raw_path}")
    return raw_path


def fit_or_load_projection(train_raw_path: Path, out_cache_root: Path, n_components: int, n_samples: int, seed: int, force: bool):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    proj_path = out_cache_root / "resnet50_pca576_scaler_pca.pkl"
    if proj_path.exists() and not force:
        with open(proj_path, "rb") as f:
            payload = pickle.load(f)
        print(f"[cache] loaded projection: {proj_path}")
        return payload["scaler"], payload["pca"]

    raw = np.load(train_raw_path, mmap_mode="r")
    flat = raw.reshape(-1, raw.shape[-1])
    rng = np.random.default_rng(seed)
    sample_n = min(n_samples, flat.shape[0])
    idx = rng.choice(flat.shape[0], size=sample_n, replace=False)
    sample = np.asarray(flat[idx], dtype=np.float32)
    scaler = StandardScaler(copy=True)
    sample_z = scaler.fit_transform(sample)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    pca.fit(sample_z)
    out_cache_root.mkdir(parents=True, exist_ok=True)
    with open(proj_path, "wb") as f:
        pickle.dump({"scaler": scaler, "pca": pca, "seed": seed, "sample_n": int(sample_n)}, f)
    print(f"[cache] fitted and saved projection: {proj_path}")
    print(f"[cache] PCA explained variance sum: {float(np.sum(pca.explained_variance_ratio_)):.4f}")
    return scaler, pca


def project_and_write_cache(
    slide_id: str,
    raw_path: Path,
    base_cache_root: Path,
    out_cache_root: Path,
    scaler,
    pca,
    batch_rows: int,
    force: bool,
) -> Path:
    out_dir = out_cache_root / slide_id
    emb_path = out_dir / "embeddings-hist.pickle"
    if emb_path.exists() and not force:
        print(f"[cache:{slide_id}] projected cache already exists: {emb_path}")
        return emb_path

    raw = np.load(raw_path, mmap_mode="r")
    h, w, c = raw.shape
    proj_path = out_dir / "resnet50_pca576_projected.npy"
    proj = np.lib.format.open_memmap(proj_path, mode="w+", dtype=np.float32, shape=(h, w, pca.n_components_))
    flat = raw.reshape(-1, c)
    proj_flat = proj.reshape(-1, pca.n_components_)
    for start in range(0, flat.shape[0], batch_rows):
        end = min(start + batch_rows, flat.shape[0])
        z = scaler.transform(np.asarray(flat[start:end], dtype=np.float32))
        proj_flat[start:end] = pca.transform(z).astype(np.float32)
        if start == 0 or (start // batch_rows) % 20 == 0:
            print(f"[cache:{slide_id}] projected rows {end}/{flat.shape[0]}", flush=True)
    proj.flush()

    img = load_base_image(base_cache_root / slide_id / "he.jpg")
    rgb = downsample_rgb(img, factor=GRID_FACTOR)
    channels = [np.asarray(proj[:, :, i], dtype=np.float32) for i in range(pca.n_components_)]
    embs = {
        "cls": channels[:192],
        "sub": channels[192:],
        "rgb": rgb,
    }
    with open(emb_path, "wb") as f:
        pickle.dump(embs, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "encoder": ENCODER_NAME,
        "source": "torchvision_resnet50_layer3_imagenet",
        "grid_factor": GRID_FACTOR,
        "pca_components": int(pca.n_components_),
        "final_channels": int(pca.n_components_ + 3),
        "shape_hw": [int(h), int(w)],
    }
    (out_dir / "encoder_ablation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[cache:{slide_id}] saved projected cache: {emb_path}")
    return emb_path


def build_resnet_cache(args: argparse.Namespace, root: Path, slides: list[str]) -> None:
    import torch

    out_cache_root = encoder_cache_root(root, args.cache_root)
    base_cache_root = root / "data" / "superres_cache"
    weights_path = args.resnet_weights or (root / "checkpoints" / "ResNet" / "resnet50-0676ba61.pth")
    if not weights_path.exists():
        raise FileNotFoundError(f"ResNet weights not found: {weights_path}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    out_cache_root.mkdir(parents=True, exist_ok=True)
    raw_paths = {}
    for sid in slides:
        raw_paths[sid] = extract_resnet_raw_features(
            sid,
            base_cache_root=base_cache_root,
            out_cache_root=out_cache_root,
            weights_path=weights_path,
            device=device,
            tile_size=args.feature_tile_size,
            force=args.force_cache,
        )

    scaler, pca = fit_or_load_projection(
        raw_paths[args.train_slide],
        out_cache_root=out_cache_root,
        n_components=args.pca_components,
        n_samples=args.pca_samples,
        seed=args.seed,
        force=args.force_cache,
    )
    for sid in slides:
        project_and_write_cache(
            sid,
            raw_paths[sid],
            base_cache_root=base_cache_root,
            out_cache_root=out_cache_root,
            scaler=scaler,
            pca=pca,
            batch_rows=args.feature_batch_rows,
            force=args.force_cache,
        )
        if not args.keep_raw_resnet:
            try:
                raw_paths[sid].unlink()
                print(f"[cache:{sid}] removed raw feature file: {raw_paths[sid]}")
            except FileNotFoundError:
                pass


def load_ablation_cache_slide(slide_id: str, root: Path, cache_root: Path, genes=None):
    import pandas as pd

    slide_dir = cache_root / slide_id
    cnts_df = pd.read_csv(slide_dir / "cnts.tsv", sep="\t", index_col=0)
    gene_names = list(cnts_df.columns)
    if genes is not None:
        cnts_df = cnts_df[genes]
        gene_names = list(cnts_df.columns)
    cnts = cnts_df.to_numpy().astype(np.float32)

    locs_df = pd.read_csv(slide_dir / "locs.tsv", sep="\t")
    spot_ids = locs_df["SpotID"].astype(str).to_list()
    locs = np.stack([locs_df["y"].to_numpy(), locs_df["x"].to_numpy()], axis=-1)
    radius = float((slide_dir / "radius.txt").read_text(encoding="utf-8").strip())

    with open(slide_dir / "embeddings-hist.pickle", "rb") as f:
        embs = pickle.load(f)
    embs = np.concatenate([embs["cls"], embs["sub"], embs["rgb"]])
    embs = embs.transpose(1, 2, 0)
    return embs, cnts, locs, radius, gene_names, spot_ids


def install_core_overrides(root: Path, cache_root: Path):
    import prostate_protocol_b_morpho as core
    from st_pipeline.superres.feature_cache import normalize_slide

    def build_slide_dataset(root_arg, slide_id, kept_genes, gene_ids, device, force_cache, max_dim, ensure_cache=True):
        emb_path = cache_root / slide_id / "embeddings-hist.pickle"
        if ensure_cache and (force_cache or not emb_path.exists()):
            raise FileNotFoundError(
                f"Ablation cache missing for {slide_id}: {emb_path}. Run --mode cache first."
            )
        embs, cnts, locs, radius, _, spot_ids = load_ablation_cache_slide(
            slide_id, root=Path(root_arg), cache_root=cache_root, genes=kept_genes
        )
        embs_norm, _, stats = normalize_slide(embs, cnts)
        ds = core.SuperresSpotDataset(embs_norm, cnts, locs, radius, gene_ids)
        meta = {
            "slide_id": slide_id,
            "embs_shape": list(embs.shape),
            "cnts_shape": list(cnts.shape),
            "locs_shape": list(locs.shape),
            "radius_px": float(radius),
            "mask_size": int(ds.mask_size),
            "embs_mean": stats["embs_mean"],
            "embs_std": stats["embs_std"],
            "spot_ids": spot_ids,
            "encoder": ENCODER_NAME,
            "cache_root": str(cache_root),
        }
        return ds, meta

    core.checkpoint_dir = checkpoint_dir
    core.result_dir = result_dir
    core.build_slide_dataset = build_slide_dataset
    return core


def core_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        force_cache=False,
        force_predict=args.force_predict,
        resume_training=args.resume_training,
        batch_size=args.batch_size,
        predict_batch_size=args.predict_batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        gene_chunk_size=args.gene_chunk_size,
        gene_lr_mult=args.gene_lr_mult,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_dim=args.max_dim,
        use_amp=args.use_amp,
        heartbeat_batches=args.heartbeat_batches,
    )


def run_train_predict(args: argparse.Namespace, root: Path, test_slides: list[str]) -> None:
    cache_root = encoder_cache_root(root, args.cache_root)
    core = install_core_overrides(root, cache_root)
    cargs = core_args(args)
    if args.mode in {"train", "all"}:
        ok = core.train_fold(cargs, root, args.train_slide, test_slides, args.tag)
        if not ok:
            return
    if args.mode in {"predict", "all"}:
        core.predict_fold(cargs, root, args.train_slide, test_slides, args.tag)
    if args.mode in {"predict", "all"}:
        core.aggregate_fold(root, args.train_slide, test_slides, args.tag)


def safe_pearson_vec(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    true = true.astype(np.float64, copy=False)
    pred = pred.astype(np.float64, copy=False)
    true_centered = true - true.mean(axis=0, keepdims=True)
    pred_centered = pred - pred.mean(axis=0, keepdims=True)
    den = np.sqrt((true_centered * true_centered).sum(axis=0) * (pred_centered * pred_centered).sum(axis=0))
    out = np.full(true.shape[1], np.nan, dtype=np.float32)
    valid = den > 0
    out[valid] = ((true_centered[:, valid] * pred_centered[:, valid]).sum(axis=0) / den[valid]).astype(np.float32)
    return out


def summarize(args: argparse.Namespace, root: Path, test_slides: list[str]) -> None:
    import pandas as pd

    out_root = result_dir(root, args.tag)
    rows = []
    hipt_dir = root / "results" / "prostate_hest" / "morpho" / "single_superres" / "INT25_to_INT28"
    if (hipt_dir / "pred_bag.npy").exists():
        pred = np.load(hipt_dir / "pred_bag.npy")
        true = np.load(hipt_dir / "true_bag.npy")
        corr = safe_pearson_vec(true, pred)
        rows.append({
            "encoder": "HIPT_pathology_FM",
            "test_slide_id": "INT28",
            "n_spots": int(pred.shape[0]),
            "n_genes": int(pred.shape[1]),
            "mean_pearson": float(np.nanmean(corr)),
            "median_pearson": float(np.nanmedian(corr)),
            "result_dir": str(hipt_dir),
        })
    for sid in test_slides:
        slide_dir = out_root / sid
        pred_path = slide_dir / "pred_bag.npy"
        true_path = slide_dir / "true_bag.npy"
        if not pred_path.exists() or not true_path.exists():
            rows.append({
                "encoder": "ImageNet_ResNet50_PCA576",
                "test_slide_id": sid,
                "status": "missing",
                "result_dir": str(slide_dir),
            })
            continue
        pred = np.load(pred_path)
        true = np.load(true_path)
        corr = safe_pearson_vec(true, pred)
        rows.append({
            "encoder": "ImageNet_ResNet50_PCA576",
            "test_slide_id": sid,
            "n_spots": int(pred.shape[0]),
            "n_genes": int(pred.shape[1]),
            "mean_pearson": float(np.nanmean(corr)),
            "median_pearson": float(np.nanmedian(corr)),
            "result_dir": str(slide_dir),
        })
    df = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    out = out_root / "pathology_encoder_ablation_summary.csv"
    df.to_csv(out, index=False)
    print(f"Saved pathology encoder summary: {out}")
    print(df)


def main() -> None:
    args = parse_args()
    root = setup_root(args.root)
    random.seed(args.seed)
    np.random.seed(args.seed)
    test_slides = split_test_slides(args.test_slides)
    slides_for_cache = sorted(set([args.train_slide] + test_slides))

    if args.mode in {"cache", "all"}:
        build_resnet_cache(args, root, slides_for_cache)
    if args.mode in {"train", "predict", "all"}:
        run_train_predict(args, root, test_slides)
    if args.mode in {"summary", "all"}:
        summarize(args, root, test_slides)


if __name__ == "__main__":
    main()
