from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import prostate_protocol_b_morpho as core  # noqa: E402


try:
    import torch
    from torch.utils.data import DataLoader
except (ModuleNotFoundError, ImportError):
    torch = None
    DataLoader = None


@dataclass(frozen=True)
class RunConfig:
    name: str
    protocol: str
    condition: str
    ckpt_dir: Path
    out_tag: str
    calibration_slides: tuple[str, ...]
    test_slides: tuple[str, ...] = ("INT28",)


@dataclass
class HiptSizeFactorEstimator:
    strategy: str
    calibration_slides: list[str]
    weights: np.ndarray
    selected_features: np.ndarray
    slope: float
    intercept: float
    clip_low: float
    clip_high: float
    top_k: int
    weight_summary: dict

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = standardize_columns(x)
        score = z @ self.weights
        log_sf = self.intercept + self.slope * score
        sf_hat_raw = np.exp(np.clip(log_sf, -8.0, 8.0)).astype(np.float32)
        sf_hat = sf_hat_raw / (float(np.mean(sf_hat_raw)) + 1e-8)
        sf_hat = np.clip(sf_hat, self.clip_low, self.clip_high)
        sf_hat = sf_hat / (float(np.mean(sf_hat)) + 1e-8)
        return sf_hat.astype(np.float32), score.astype(np.float32), log_sf.astype(np.float32)

    def metadata(self) -> dict:
        payload = asdict(self)
        payload["weights"] = {
            "n_features": int(self.weights.shape[0]),
            "nonzero": int(np.count_nonzero(self.weights)),
            "l1_norm": float(np.sum(np.abs(self.weights))),
            "l2_norm": float(np.sqrt(np.sum(self.weights**2))),
        }
        payload["selected_features"] = self.selected_features.astype(int).tolist()
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse existing prostate CellFM checkpoints, predict normalized rates, "
            "and replace unavailable test size factors with HIPT-derived size-factor estimates."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--run", choices=["all", "single_pretrained", "single_random", "multi_pretrained", "multi_random"], default="all")
    parser.add_argument("--out-root-name", default="cellfm_hipt_sizefactor_ablation_v1")
    parser.add_argument("--predict-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gene-chunk-size", type=int, default=512)
    parser.add_argument("--max-dim", type=int, default=16000)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-predict", action="store_true")
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--heartbeat-batches", type=int, default=25)
    parser.add_argument("--sf-top-k", type=int, default=64)
    parser.add_argument("--sf-clip-low-quantile", type=float, default=0.01)
    parser.add_argument("--sf-clip-high-quantile", type=float, default=0.99)
    return parser.parse_args()


def setup_root(root_arg: Path | None) -> Path:
    root = root_arg or Path(os.environ.get("MORPHO_VC_ROOT", "."))
    root = root.expanduser().resolve()
    os.chdir(root)
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.append(str(src))
    return root


def get_configs(root: Path) -> dict[str, RunConfig]:
    base = root / "checkpoints" / "st_superres"
    return {
        "single_pretrained": RunConfig(
            name="single_pretrained",
            protocol="single",
            condition="Pretrained CellFM + HIPT-estimated size factor",
            ckpt_dir=base / "cellfm_single_INT25_to_INT28",
            out_tag="single_pretrained_INT28",
            calibration_slides=("INT25",),
        ),
        "single_random": RunConfig(
            name="single_random",
            protocol="single",
            condition="Random CellFM + HIPT-estimated size factor",
            ckpt_dir=base / "cellfm_randominit_ablation" / "single_seed42",
            out_tag="single_random_INT28",
            calibration_slides=("INT25",),
        ),
        "multi_pretrained": RunConfig(
            name="multi_pretrained",
            protocol="multi",
            condition="Pretrained CellFM + HIPT-estimated size factor",
            ckpt_dir=base / "cellfm_forward_sum",
            out_tag="multi_pretrained_INT28",
            calibration_slides=("INT25", "INT26", "INT27"),
        ),
        "multi_random": RunConfig(
            name="multi_random",
            protocol="multi",
            condition="Random CellFM + HIPT-estimated size factor",
            ckpt_dir=base / "cellfm_randominit_ablation" / "multi_seed42",
            out_tag="multi_random_INT28",
            calibration_slides=("INT25", "INT26", "INT27"),
        ),
    }


def result_root(root: Path, args: argparse.Namespace, cfg: RunConfig) -> Path:
    return root / "results" / "st_superres" / args.out_root_name / cfg.out_tag


def require_best_model(cfg: RunConfig) -> Path:
    ckpt = cfg.ckpt_dir / "best_model.pt"
    if not ckpt.exists() or ckpt.stat().st_size == 0:
        raise FileNotFoundError(f"Missing checkpoint for {cfg.name}: {ckpt}")
    return ckpt


def standardize_columns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = np.nanmean(x, axis=0, keepdims=True)
    std = np.nanstd(x, axis=0, keepdims=True)
    z = (x - mean) / (std + 1e-6)
    z[~np.isfinite(z)] = 0.0
    return z.astype(np.float32)


def pearson_columns(x_z: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    y_z = (y - float(np.mean(y))) / (float(np.std(y)) + 1e-6)
    corr = (x_z.astype(np.float64).T @ y_z.astype(np.float64)) / max(x_z.shape[0] - 1, 1)
    corr[~np.isfinite(corr)] = 0.0
    return corr.astype(np.float32)


def read_raw_spot_features(root: Path, slide_id: str, kept_genes: list[str], gene_ids) -> tuple[np.ndarray, np.ndarray, list[str]]:
    from st_pipeline.superres.feature_cache import load_cache_slide

    embs, cnts, locs, radius, _, spot_ids = load_cache_slide(slide_id, root=root, genes=kept_genes)
    ds = core.SuperresSpotDataset(embs, cnts, locs, radius, gene_ids)
    mask = ds.mask
    offsets = np.argwhere(mask).astype(np.int64) - (np.array(mask.shape, dtype=np.int64) // 2)
    yy = ds.locs[:, [0]] + offsets[:, 0][None, :]
    xx = ds.locs[:, [1]] + offsets[:, 1][None, :]
    x = np.nanmean(embs[yy, xx, :], axis=1).astype(np.float32)
    cnts_valid = cnts[ds.keep_idx]
    totals = cnts_valid.sum(axis=1).astype(np.float32)
    sf = totals / (float(np.mean(totals)) + 1e-8)
    return x, sf.astype(np.float32), [str(spot_ids[i]) for i in ds.keep_idx]


def fit_hipt_size_factor_estimator(
    root: Path,
    kept_genes: list[str],
    gene_ids,
    calibration_slides: tuple[str, ...],
    args: argparse.Namespace,
    out_dir: Path,
) -> HiptSizeFactorEstimator:
    slide_corrs = []
    x_blocks = []
    y_blocks = []
    slide_rows = []
    for slide in calibration_slides:
        x, sf, _ = read_raw_spot_features(root, slide, kept_genes, gene_ids)
        y = np.log(np.clip(sf, 1e-6, None)).astype(np.float32)
        z = standardize_columns(x)
        corr = pearson_columns(z, y)
        slide_corrs.append(corr)
        x_blocks.append(z)
        y_blocks.append(y)
        abs_corr = np.abs(corr)
        slide_rows.append(
            {
                "slide": slide,
                "n_spots": int(x.shape[0]),
                "n_features": int(x.shape[1]),
                "median_abs_r": float(np.median(abs_corr)),
                "p95_abs_r": float(np.percentile(abs_corr, 95)),
                "max_abs_r": float(np.max(abs_corr)),
            }
        )

    corr_stack = np.vstack(slide_corrs)
    fisher = np.arctanh(np.clip(corr_stack, -0.999, 0.999))
    weights = np.tanh(np.nanmedian(fisher, axis=0)).astype(np.float32)
    weights[~np.isfinite(weights)] = 0.0
    top_k = min(int(args.sf_top_k), weights.shape[0])
    selected = np.argsort(np.abs(weights))[::-1][:top_k]
    pruned = np.zeros_like(weights, dtype=np.float32)
    pruned[selected] = weights[selected]
    norm = float(np.sum(np.abs(pruned)))
    if norm <= 0:
        raise ValueError("HIPT size-factor weights are all zero.")
    pruned /= norm

    x_cal = np.concatenate(x_blocks, axis=0)
    y_cal = np.concatenate(y_blocks, axis=0)
    score = x_cal @ pruned
    slope, intercept = np.polyfit(score.astype(np.float64), y_cal.astype(np.float64), deg=1)
    sf_cal = np.exp(y_cal)
    clip_low = float(np.quantile(sf_cal, args.sf_clip_low_quantile))
    clip_high = float(np.quantile(sf_cal, args.sf_clip_high_quantile))

    estimator = HiptSizeFactorEstimator(
        strategy="median_fisher_correlation_topk_affine_logsf",
        calibration_slides=list(calibration_slides),
        weights=pruned.astype(np.float32),
        selected_features=selected.astype(np.int64),
        slope=float(slope),
        intercept=float(intercept),
        clip_low=clip_low,
        clip_high=clip_high,
        top_k=top_k,
        weight_summary={
            "slide_summaries": slide_rows,
            "median_abs_weight": float(np.median(np.abs(weights))),
            "p95_abs_weight": float(np.percentile(np.abs(weights), 95)),
            "max_abs_weight": float(np.max(np.abs(weights))),
            "calibration_score_pearson_logsf": float(np.corrcoef(score, y_cal)[0, 1]),
        },
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "hipt_sf_weights.npy", estimator.weights)
    pd.DataFrame(slide_rows).to_csv(out_dir / "hipt_sf_calibration_slide_correlations.csv", index=False)
    (out_dir / "hipt_sf_estimator_metadata.json").write_text(json.dumps(estimator.metadata(), indent=2), encoding="utf-8")
    return estimator


def predict_rate_chunk(model, batch: dict[str, torch.Tensor], gene_ids: torch.Tensor) -> torch.Tensor:
    from st_pipeline.constants import KEYS

    x = batch[KEYS.X]
    ptr = batch[KEYS.PTR_BAG_INSTANCE]
    if gene_ids.dim() == 1:
        gene_ids = gene_ids.unsqueeze(0).expand(x.size(0), -1)
    prompt = model.adapter(x)
    cell_emb, gene_emb = model.cellfm(prompt, gene_ids)
    pred = model.cellfm.cellwise_decode(cell_emb, gene_emb)
    mu_inst = model.softplus(pred)
    num_bags = int(ptr.max().item()) + 1
    return model._aggregate(mu_inst, ptr, num_bags)


def evaluate_sf(sf_true: np.ndarray, sf_hat: np.ndarray, score: np.ndarray) -> dict:
    sf_true = np.asarray(sf_true, dtype=np.float64)
    sf_hat = np.asarray(sf_hat, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    pearson = float(np.corrcoef(sf_true, sf_hat)[0, 1])
    log_pearson = float(np.corrcoef(np.log(np.clip(sf_true, 1e-6, None)), np.log(np.clip(sf_hat, 1e-6, None)))[0, 1])
    score_pearson = float(np.corrcoef(np.log(np.clip(sf_true, 1e-6, None)), score)[0, 1])
    return {
        "sf_pearson": pearson,
        "log_sf_pearson": log_pearson,
        "score_log_sf_pearson": score_pearson,
        "sf_mae": float(np.mean(np.abs(sf_hat - sf_true))),
        "sf_rmse": float(np.sqrt(np.mean((sf_hat - sf_true) ** 2))),
        "sf_hat_min": float(np.min(sf_hat)),
        "sf_hat_max": float(np.max(sf_hat)),
        "sf_true_min": float(np.min(sf_true)),
        "sf_true_max": float(np.max(sf_true)),
    }


def write_metadata(slide_out: Path, cfg: RunConfig, ckpt_path: Path, estimator: HiptSizeFactorEstimator, sf_metrics: dict) -> None:
    meta = {
        "run": cfg.name,
        "protocol": cfg.protocol,
        "condition": cfg.condition,
        "source_checkpoint": str(ckpt_path),
        "prediction_target": "count_scale_from_normalized_rate_times_hipt_estimated_size_factor",
        "pred_bag_file": "pred_bag.npy",
        "true_bag_file": "true_bag.npy",
        "pred_rate_file": "pred_rate_bag.npy",
        "true_rate_file": "true_rate_bag.npy",
        "sf_hat_file": "sf_hat.npy",
        "sf_true_file": "sf_true.npy",
        "uses_test_size_factor_for_prediction": False,
        "uses_test_size_factor_for_sizefactor_weight_training": False,
        "uses_test_size_factor_for_label_and_diagnostics": True,
        "estimator": estimator.metadata(),
        "sf_metrics_on_test": sf_metrics,
    }
    (slide_out / "hipt_sizefactor_inference_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def predict_config(args: argparse.Namespace, root: Path, cfg: RunConfig) -> list[dict]:
    if torch is None:
        raise ModuleNotFoundError("torch is required. Run with conda env morpho-vc/scellst-bench.")

    from st_pipeline.constants import KEYS
    from st_pipeline.data.collate import mil_collate

    ckpt_path = require_best_model(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    kept_genes, gene_ids, vocab = core.load_gene_setup(root, cfg.ckpt_dir, write_metadata=False)
    chunks = core.gene_chunks(len(kept_genes), args.gene_chunk_size)
    out_root = result_root(root, args, cfg)
    out_root.mkdir(parents=True, exist_ok=True)
    estimator = fit_hipt_size_factor_estimator(root, kept_genes, gene_ids, cfg.calibration_slides, args, out_root)

    print(f"[{cfg.name}] checkpoint={ckpt_path}")
    print(f"[{cfg.name}] output={out_root}")
    print(f"[{cfg.name}] calibration_slides={','.join(cfg.calibration_slides)}")
    print(f"[{cfg.name}] genes={len(kept_genes)} chunks={len(chunks)} device={device}")
    print(f"[{cfg.name}] sf calibration score r={estimator.weight_summary['calibration_score_pearson_logsf']:.4f}")

    rows = []
    model = None
    input_dim = None
    for sid in cfg.test_slides:
        slide_out = out_root / sid
        pred_path = slide_out / "pred_bag.npy"
        true_path = slide_out / "true_bag.npy"
        if pred_path.exists() and true_path.exists() and not args.force_predict:
            pred = np.load(pred_path)
            true = np.load(true_path)
            metrics = core.save_metrics(pred, true, kept_genes, slide_out)
            rows.append({"run": cfg.name, "protocol": cfg.protocol, "condition": cfg.condition, "test_slide_id": sid, "result_dir": str(slide_out), **metrics})
            continue

        t0 = time.perf_counter()
        x_raw, sf_true_diag, spot_ids_raw = read_raw_spot_features(root, sid, kept_genes, gene_ids)
        sf_hat, sf_score, sf_log_hat = estimator.predict(x_raw)
        sf_metrics = evaluate_sf(sf_true_diag, sf_hat, sf_score)
        ds, meta = core.build_slide_dataset(root, sid, kept_genes, gene_ids, device, args.force_cache, args.max_dim, ensure_cache=True)
        if len(ds) != len(sf_hat):
            raise ValueError(f"Dataset/size-factor length mismatch for {sid}: {len(ds)} vs {len(sf_hat)}")

        loader = DataLoader(
            ds,
            batch_size=args.predict_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=mil_collate,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )

        if input_dim is None:
            sample_batch = next(iter(loader))
            input_dim = int(sample_batch[KEYS.X].shape[-1])
            model = core.build_model(input_dim, len(kept_genes), len(vocab), root, device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
        assert model is not None

        pred_rate = np.zeros((len(ds), len(kept_genes)), dtype=np.float32)
        pred_count = np.zeros((len(ds), len(kept_genes)), dtype=np.float32)
        true_rate = np.zeros((len(ds), len(kept_genes)), dtype=np.float32)
        true_count = np.zeros((len(ds), len(kept_genes)), dtype=np.float32)
        offset = 0
        use_amp = bool(args.use_amp and device.type == "cuda")
        with torch.inference_mode():
            for b_idx, batch in enumerate(loader):
                for key in batch:
                    batch[key] = batch[key].to(device, non_blocking=True)
                batch_size = int(batch[KEYS.Y_BAG].shape[0])
                true_count[offset : offset + batch_size] = batch[KEYS.Y_BAG].float().cpu().numpy()
                sf_true_batch = batch[KEYS.SIZE_FACTOR].float().clamp_min(1e-8).unsqueeze(1)
                true_rate[offset : offset + batch_size] = (batch[KEYS.Y_BAG].float() / sf_true_batch).cpu().numpy()
                sf_hat_batch = sf_hat[offset : offset + batch_size, None]

                chunk_rate_preds = []
                chunk_count_preds = []
                for chunk in chunks:
                    idx = torch.as_tensor(chunk, device=device)
                    chunk_gene_ids = batch[KEYS.GENE_IDS][idx]
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        mu_rate_bag = predict_rate_chunk(model, batch, chunk_gene_ids)
                    mu_rate_np = mu_rate_bag.float().cpu().numpy()
                    chunk_rate_preds.append(mu_rate_np)
                    chunk_count_preds.append(mu_rate_np * sf_hat_batch)
                pred_rate[offset : offset + batch_size] = np.concatenate(chunk_rate_preds, axis=1)
                pred_count[offset : offset + batch_size] = np.concatenate(chunk_count_preds, axis=1)
                offset += batch_size
                if b_idx % args.heartbeat_batches == 0:
                    print(f"[{cfg.name}:{sid}] hipt-sf predict batch {b_idx}/{len(loader)}", flush=True)

        slide_out.mkdir(parents=True, exist_ok=True)
        np.save(slide_out / "pred_bag.npy", pred_count)
        np.save(slide_out / "true_bag.npy", true_count)
        np.save(slide_out / "pred_rate_bag.npy", pred_rate)
        np.save(slide_out / "true_rate_bag.npy", true_rate)
        np.save(slide_out / "sf_hat.npy", sf_hat)
        np.save(slide_out / "sf_true.npy", sf_true_diag.astype(np.float32))
        np.save(slide_out / "sf_score.npy", sf_score)
        np.save(slide_out / "sf_log_hat.npy", sf_log_hat)
        spot_ids = np.asarray(meta["spot_ids"], dtype=str)[ds.keep_idx]
        (slide_out / "spot_ids.txt").write_text("\n".join(map(str, spot_ids)), encoding="utf-8")
        if list(spot_ids) != spot_ids_raw:
            (slide_out / "spot_id_alignment_warning.txt").write_text("Raw and normalized dataset spot IDs differed.\n", encoding="utf-8")
        count_metrics = core.save_metrics(pred_count, true_count, kept_genes, slide_out)
        rate_metrics = core.save_metrics(pred_rate, true_rate, kept_genes, slide_out / "rate_metrics")
        (slide_out / "sf_estimator_metrics.json").write_text(json.dumps(sf_metrics, indent=2), encoding="utf-8")
        write_metadata(slide_out, cfg, ckpt_path, estimator, sf_metrics)

        row = {
            "run": cfg.name,
            "protocol": cfg.protocol,
            "condition": cfg.condition,
            "test_slide_id": sid,
            "result_dir": str(slide_out),
            **count_metrics,
            "Rate_Mean_Pearson": rate_metrics["Mean_Pearson"],
            "Rate_HVG1000_Mean_Pearson": rate_metrics["HVG1000_Mean_Pearson"],
            **{f"SF_{k}": v for k, v in sf_metrics.items()},
        }
        rows.append(row)
        print(
            f"[{cfg.name}:{sid}] done in {time.perf_counter() - t0:.1f}s | "
            f"count mean={count_metrics['Mean_Pearson']:.4f} "
            f"rate mean={rate_metrics['Mean_Pearson']:.4f} sf_r={sf_metrics['sf_pearson']:.4f}",
            flush=True,
        )

    pd.DataFrame(rows).to_csv(out_root / "fold_summary_hipt_sizefactor.csv", index=False)
    return rows


def main() -> None:
    args = parse_args()
    root = setup_root(args.root)
    configs = get_configs(root)
    selected = list(configs) if args.run == "all" else [args.run]
    rows = []
    for name in selected:
        rows.extend(predict_config(args, root, configs[name]))
    summary = pd.DataFrame(rows)
    summary_root = root / "results" / "st_superres" / args.out_root_name
    summary_root.mkdir(parents=True, exist_ok=True)
    out = summary_root / "cellfm_hipt_sizefactor_key_results.csv"
    summary.to_csv(out, index=False)
    print(summary)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
