from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
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
    test_slides: tuple[str, ...] = ("INT28",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean normalized-rate inference for prostate CellFM pretrained and "
            "random-init branches. Existing checkpoints are reused; test spot "
            "size factors are not used in the prediction path."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--run", choices=["all", "single_pretrained", "single_random", "multi_pretrained", "multi_random"], default="all")
    parser.add_argument("--predict-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gene-chunk-size", type=int, default=1024)
    parser.add_argument("--max-dim", type=int, default=16000)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-predict", action="store_true")
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--heartbeat-batches", type=int, default=25)
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
            condition="Pretrained CellFM",
            ckpt_dir=base / "cellfm_single_INT25_to_INT28",
            out_tag="single_pretrained_INT28",
        ),
        "single_random": RunConfig(
            name="single_random",
            protocol="single",
            condition="Random CellFM",
            ckpt_dir=base / "cellfm_randominit_ablation" / "single_seed42",
            out_tag="single_random_INT28",
        ),
        "multi_pretrained": RunConfig(
            name="multi_pretrained",
            protocol="multi",
            condition="Pretrained CellFM",
            ckpt_dir=base / "cellfm_forward_sum",
            out_tag="multi_pretrained_INT28",
        ),
        "multi_random": RunConfig(
            name="multi_random",
            protocol="multi",
            condition="Random CellFM",
            ckpt_dir=base / "cellfm_randominit_ablation" / "multi_seed42",
            out_tag="multi_random_INT28",
        ),
    }


def clean_result_root(root: Path, cfg: RunConfig) -> Path:
    return root / "results" / "st_superres" / "cellfm_clean_rate_ablation" / cfg.out_tag


def require_best_model(cfg: RunConfig) -> Path:
    ckpt = cfg.ckpt_dir / "best_model.pt"
    if not ckpt.exists() or ckpt.stat().st_size == 0:
        raise FileNotFoundError(f"Missing checkpoint for {cfg.name}: {ckpt}")
    return ckpt


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


def write_metadata(slide_out: Path, cfg: RunConfig, ckpt_path: Path) -> None:
    meta = {
        "run": cfg.name,
        "protocol": cfg.protocol,
        "condition": cfg.condition,
        "source_checkpoint": str(ckpt_path),
        "prediction_target": "normalized_expression_rate",
        "prediction_file": "pred_rate_bag.npy",
        "ground_truth_file": "true_rate_bag.npy",
        "uses_test_size_factor_for_prediction": False,
        "uses_test_size_factor_for_label_normalization": True,
        "checkpoint_reused": True,
    }
    (slide_out / "clean_rate_inference_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def predict_config(args: argparse.Namespace, root: Path, cfg: RunConfig) -> list[dict]:
    if torch is None:
        raise ModuleNotFoundError("torch is required. Run with conda env scellst_bench.")

    from st_pipeline.constants import KEYS
    from st_pipeline.data.collate import mil_collate

    ckpt_path = require_best_model(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    kept_genes, gene_ids, vocab = core.load_gene_setup(root, cfg.ckpt_dir, write_metadata=False)
    chunks = core.gene_chunks(len(kept_genes), args.gene_chunk_size)
    out_root = clean_result_root(root, cfg)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[{cfg.name}] checkpoint={ckpt_path}")
    print(f"[{cfg.name}] output={out_root}")
    print(f"[{cfg.name}] genes={len(kept_genes)} chunks={len(chunks)} device={device}")

    rows = []
    model = None
    input_dim = None
    for sid in cfg.test_slides:
        slide_out = out_root / sid
        pred_path = slide_out / "pred_rate_bag.npy"
        true_path = slide_out / "true_rate_bag.npy"
        if pred_path.exists() and true_path.exists() and not args.force_predict:
            pred_rate = np.load(pred_path)
            true_rate = np.load(true_path)
            metrics = core.save_metrics(pred_rate, true_rate, kept_genes, slide_out)
            rows.append({"run": cfg.name, "protocol": cfg.protocol, "condition": cfg.condition, "test_slide_id": sid, "result_dir": str(slide_out), **metrics})
            continue

        t0 = time.perf_counter()
        ds, meta = core.build_slide_dataset(
            root,
            sid,
            kept_genes,
            gene_ids,
            device,
            args.force_cache,
            args.max_dim,
            ensure_cache=True,
        )
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
        true_rate = np.zeros((len(ds), len(kept_genes)), dtype=np.float32)
        offset = 0
        use_amp = bool(args.use_amp and device.type == "cuda")
        with torch.inference_mode():
            for b_idx, batch in enumerate(loader):
                for key in batch:
                    batch[key] = batch[key].to(device, non_blocking=True)
                batch_size = int(batch[KEYS.Y_BAG].shape[0])
                sf = batch[KEYS.SIZE_FACTOR].float().clamp_min(1e-8).unsqueeze(1)
                true_rate[offset : offset + batch_size] = (batch[KEYS.Y_BAG].float() / sf).cpu().numpy()

                chunk_preds = []
                for chunk in chunks:
                    idx = torch.as_tensor(chunk, device=device)
                    chunk_gene_ids = batch[KEYS.GENE_IDS][idx]
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        mu_rate_bag = predict_rate_chunk(model, batch, chunk_gene_ids)
                    chunk_preds.append(mu_rate_bag.float().cpu().numpy())
                pred_rate[offset : offset + batch_size] = np.concatenate(chunk_preds, axis=1)
                offset += batch_size
                if b_idx % args.heartbeat_batches == 0:
                    print(f"[{cfg.name}:{sid}] clean-rate predict batch {b_idx}/{len(loader)}", flush=True)

        slide_out.mkdir(parents=True, exist_ok=True)
        np.save(pred_path, pred_rate)
        np.save(true_path, true_rate)
        np.save(slide_out / "pred_bag.npy", pred_rate)
        np.save(slide_out / "true_bag.npy", true_rate)
        spot_ids = np.asarray(meta["spot_ids"], dtype=str)[ds.keep_idx]
        (slide_out / "spot_ids.txt").write_text("\n".join(map(str, spot_ids)), encoding="utf-8")
        write_metadata(slide_out, cfg, ckpt_path)
        metrics = core.save_metrics(pred_rate, true_rate, kept_genes, slide_out)
        rows.append({"run": cfg.name, "protocol": cfg.protocol, "condition": cfg.condition, "test_slide_id": sid, "result_dir": str(slide_out), **metrics})
        print(f"[{cfg.name}:{sid}] done in {time.perf_counter() - t0:.1f}s | rate mean={metrics['Mean_Pearson']:.4f}")

    pd.DataFrame(rows).to_csv(out_root / "fold_summary_clean_rate.csv", index=False)
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
    out_path = root / "results" / "st_superres" / "cellfm_clean_rate_ablation" / "cellfm_clean_rate_key_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    keep = [
        "run",
        "protocol",
        "condition",
        "test_slide_id",
        "Mean_Pearson",
        "Median_Pearson",
        "HVG1000_Mean_Pearson",
        "HVG1000_Median_Pearson",
        "Best_Gene",
        "Best_Pearson",
    ]
    print(summary[keep].to_string(index=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
