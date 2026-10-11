from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CONFIGS = {
    "single": {
        "mlp_rate_dir": ROOT / "results" / "ablation_decoder" / "mlp_decoder_clean_rate_single_INT25_to_INT28" / "INT28",
        "sf_dir": ROOT / "results" / "st_superres" / "cellfm_hipt_sizefactor_ablation_v1" / "single_pretrained_INT28" / "INT28",
        "out_dir": ROOT / "results" / "ablation_decoder" / "mlp_decoder_hipt_sizefactor_single_INT25_to_INT28" / "INT28",
        "out_root": ROOT / "results" / "ablation_decoder" / "mlp_decoder_hipt_sizefactor_single_INT25_to_INT28",
        "protocol": "single",
        "calibration_slides": ["INT25"],
    },
    "multi": {
        "mlp_rate_dir": ROOT / "results" / "ablation_decoder" / "mlp_decoder_clean_rate_multi_INT25_INT26_val_INT27_to_INT28" / "INT28",
        "sf_dir": ROOT / "results" / "st_superres" / "cellfm_hipt_sizefactor_ablation_v1" / "multi_pretrained_INT28" / "INT28",
        "out_dir": ROOT / "results" / "ablation_decoder" / "mlp_decoder_hipt_sizefactor_multi_INT25_INT26_val_INT27_to_INT28" / "INT28",
        "out_root": ROOT / "results" / "ablation_decoder" / "mlp_decoder_hipt_sizefactor_multi_INT25_INT26_val_INT27_to_INT28",
        "protocol": "multi",
        "calibration_slides": ["INT25", "INT26", "INT27"],
    },
}


def load_genes(n_genes: int) -> list[str]:
    candidates = [
        ROOT / "checkpoints" / "st_superres" / "decoder_ablation" / "mlp_decoder_single_INT25_to_INT28" / "common_genes.json",
        ROOT / "checkpoints" / "st_superres" / "decoder_ablation" / "mlp_decoder_multi_INT25_INT26_val_INT27_to_INT28" / "common_genes.json",
        ROOT / "checkpoints" / "st_superres" / "cellfm_forward_sum" / "common_genes.json",
    ]
    for path in candidates:
        if path.exists():
            genes = json.loads(path.read_text(encoding="utf-8"))
            if len(genes) == n_genes:
                return [str(g) for g in genes]
    raise ValueError(f"No gene list found for {n_genes} genes")


def pearson_per_gene(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    true = true.astype(np.float64, copy=False)
    pred = pred.astype(np.float64, copy=False)
    true_centered = true - true.mean(axis=0, keepdims=True)
    pred_centered = pred - pred.mean(axis=0, keepdims=True)
    num = np.sum(true_centered * pred_centered, axis=0)
    den = np.sqrt(np.sum(true_centered**2, axis=0) * np.sum(pred_centered**2, axis=0))
    corr = np.full(true.shape[1], np.nan, dtype=np.float32)
    valid = den > 0
    corr[valid] = (num[valid] / den[valid]).astype(np.float32)
    return corr


def save_metrics(pred: np.ndarray, true: np.ndarray, genes: list[str], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    corr_all = pearson_per_gene(true, pred)
    var_true = np.var(true, axis=0)
    var_pred = np.var(pred, axis=0)
    top_hvg = min(1000, true.shape[1])
    hvg_idx_true = np.argsort(var_true)[::-1][:top_hvg]
    hvg_idx_pred = np.argsort(var_pred)[::-1][:top_hvg]
    overlap_idx = np.array(sorted(set(hvg_idx_true) & set(hvg_idx_pred)), dtype=np.int64)
    corr_hvg = corr_all[hvg_idx_true]
    corr_overlap = corr_all[overlap_idx] if overlap_idx.size else np.array([], dtype=np.float32)
    valid = np.where(np.isfinite(corr_all))[0]
    best_idx = int(valid[np.nanargmax(corr_all[valid])]) if valid.size else None
    metrics = {
        "n_spots": int(pred.shape[0]),
        "n_genes": int(pred.shape[1]),
        "MAE": float(np.mean(np.abs(pred - true))),
        "RMSE": float(np.sqrt(np.mean((pred - true) ** 2))),
        "Mean_Pearson": float(np.nanmean(corr_all)),
        "Median_Pearson": float(np.nanmedian(corr_all)),
        "HVG1000_Mean_Pearson": float(np.nanmean(corr_hvg)),
        "HVG1000_Median_Pearson": float(np.nanmedian(corr_hvg)),
        "HVG1000_Overlap": int(overlap_idx.size),
        "Best_Gene": genes[best_idx] if best_idx is not None else "NA",
        "Best_Pearson": float(corr_all[best_idx]) if best_idx is not None else None,
    }
    np.save(out_dir / "gene_corr_all.npy", corr_all)
    np.save(out_dir / "gene_corr_hvg.npy", corr_hvg)
    np.save(out_dir / "hvg_idx_true.npy", hvg_idx_true)
    np.save(out_dir / "hvg_idx_pred.npy", hvg_idx_pred)
    np.save(out_dir / "corr_overlap_genes.npy", corr_overlap)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def ensure_same_spots(left: Path, right: Path) -> None:
    left_ids = (left / "spot_ids.txt").read_text(encoding="utf-8").splitlines()
    right_ids = (right / "spot_ids.txt").read_text(encoding="utf-8").splitlines()
    if left_ids != right_ids:
        raise ValueError(f"Spot order differs: {left} vs {right}")


def apply_protocol(name: str, cfg: dict) -> dict:
    rate_dir = cfg["mlp_rate_dir"]
    sf_dir = cfg["sf_dir"]
    out_dir = cfg["out_dir"]
    out_root = cfg["out_root"]
    ensure_same_spots(rate_dir, sf_dir)

    pred_rate = np.load(rate_dir / "pred_bag.npy").astype(np.float32)
    true_rate = np.load(rate_dir / "true_bag.npy").astype(np.float32)
    sf_hat = np.load(sf_dir / "sf_hat.npy").astype(np.float32)
    sf_true = np.load(sf_dir / "sf_true.npy").astype(np.float32)
    true_count = np.load(sf_dir / "true_bag.npy").astype(np.float32)
    pred_count = pred_rate * sf_hat[:, None]

    genes = load_genes(pred_count.shape[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "pred_bag.npy", pred_count)
    np.save(out_dir / "true_bag.npy", true_count)
    np.save(out_dir / "pred_rate_bag.npy", pred_rate)
    np.save(out_dir / "true_rate_bag.npy", true_rate)
    np.save(out_dir / "sf_hat.npy", sf_hat)
    np.save(out_dir / "sf_true.npy", sf_true)
    shutil.copy2(rate_dir / "spot_ids.txt", out_dir / "spot_ids.txt")
    for fname in ["sf_estimator_metrics.json", "hipt_sizefactor_inference_metadata.json"]:
        src = sf_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)

    count_metrics = save_metrics(pred_count, true_count, genes, out_dir)
    rate_metrics = save_metrics(pred_rate, true_rate, genes, out_dir / "rate_metrics")
    sf_metrics = json.loads((sf_dir / "sf_estimator_metrics.json").read_text(encoding="utf-8"))
    meta = {
        "protocol": cfg["protocol"],
        "decoder": "MLP decoder",
        "source_rate_dir": str(rate_dir),
        "source_hipt_sizefactor_dir": str(sf_dir),
        "calibration_slides": cfg["calibration_slides"],
        "uses_test_size_factor_for_prediction": False,
        "prediction_target": "count_scale_from_mlp_rate_times_hipt_estimated_size_factor",
    }
    (out_dir / "mlp_hipt_sizefactor_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    row = {
        "run": f"{name}_mlp",
        "protocol": cfg["protocol"],
        "condition": "MLP decoder + HIPT-estimated size factor",
        "test_slide_id": "INT28",
        "result_dir": str(out_dir),
        **count_metrics,
        "Rate_Mean_Pearson": rate_metrics["Mean_Pearson"],
        "Rate_HVG1000_Mean_Pearson": rate_metrics["HVG1000_Mean_Pearson"],
        **{f"SF_{k}": v for k, v in sf_metrics.items()},
    }
    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out_root / "fold_summary_hipt_sizefactor.csv", index=False)
    return row


def main() -> None:
    rows = [apply_protocol(name, cfg) for name, cfg in CONFIGS.items()]
    out = ROOT / "results" / "ablation_decoder" / "mlp_decoder_hipt_sizefactor_key_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
