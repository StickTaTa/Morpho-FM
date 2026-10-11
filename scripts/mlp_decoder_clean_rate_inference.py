from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import ablation_mlp_decoder_prostate as mlp_ablation  # noqa: E402
import prostate_protocol_b_morpho as core  # noqa: E402


try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except (ModuleNotFoundError, ImportError):
    torch = None
    F = None
    DataLoader = None


DEFAULT_SINGLE_SOURCE_TAG = "mlp_decoder_single_INT25_to_INT28"
DEFAULT_MULTI_SOURCE_TAG = "mlp_decoder_multi_INT25_INT26_val_INT27_to_INT28"
DEFAULT_SINGLE_OUTPUT_TAG = "mlp_decoder_clean_rate_single_INT25_to_INT28"
DEFAULT_MULTI_OUTPUT_TAG = "mlp_decoder_clean_rate_multi_INT25_INT26_val_INT27_to_INT28"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run clean normalized-rate inference for the MLP decoder ablation. "
            "The trained checkpoint is reused, and test-slide size factors are "
            "not used by the model prediction path."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--protocol", choices=["single", "multi", "both"], default="both")
    parser.add_argument("--single-source-tag", default=DEFAULT_SINGLE_SOURCE_TAG)
    parser.add_argument("--multi-source-tag", default=DEFAULT_MULTI_SOURCE_TAG)
    parser.add_argument("--single-output-tag", default=DEFAULT_SINGLE_OUTPUT_TAG)
    parser.add_argument("--multi-output-tag", default=DEFAULT_MULTI_OUTPUT_TAG)
    parser.add_argument("--single-test-slides", default="INT28")
    parser.add_argument("--multi-test-slides", default="INT28")
    parser.add_argument("--predict-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gene-chunk-size", type=int, default=1024)
    parser.add_argument("--max-dim", type=int, default=16000)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-predict", action="store_true")
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--heartbeat-batches", type=int, default=5)
    return parser.parse_args()


def setup_root(root_arg: Path | None) -> Path:
    root = root_arg or Path(os.environ.get("MORPHO_VC_ROOT", "."))
    root = root.expanduser().resolve()
    os.chdir(root)
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.append(str(src))
    return root


def result_dir(root: Path, tag: str) -> Path:
    return root / "results" / "ablation_decoder" / tag


def load_genes(root: Path, source_tag: str) -> tuple[list[str], np.ndarray]:
    from st_pipeline.data.gene_vocab import load_gene_vocab, map_genes_to_vocab

    ckpt_dir = mlp_ablation.checkpoint_dir(root, source_tag)
    kept_genes = json.loads((ckpt_dir / "common_genes.json").read_text(encoding="utf-8"))
    vocab = load_gene_vocab(root / "assets" / "cellfm" / "expand_gene_info.csv")
    kept_genes, gene_ids = map_genes_to_vocab(kept_genes, vocab)
    return kept_genes, gene_ids


def predict_rate_chunk(model, batch: dict[str, torch.Tensor], gene_indices: torch.Tensor) -> torch.Tensor:
    from st_pipeline.constants import KEYS

    x = batch[KEYS.X]
    ptr = batch[KEYS.PTR_BAG_INSTANCE]
    h = model.adapter(x)
    pred = F.linear(h, model.decoder.weight[gene_indices], model.decoder.bias[gene_indices])
    mu_inst = model.softplus(pred)
    num_bags = int(ptr.max().item()) + 1
    return model._aggregate(mu_inst, ptr, num_bags)


def write_metadata(out_dir: Path, source_tag: str, output_tag: str, slide_id: str) -> None:
    meta = {
        "source_checkpoint_tag": source_tag,
        "output_tag": output_tag,
        "test_slide_id": slide_id,
        "prediction_target": "normalized_expression_rate",
        "prediction_file": "pred_rate_bag.npy",
        "ground_truth_file": "true_rate_bag.npy",
        "uses_test_size_factor_for_prediction": False,
        "uses_test_size_factor_for_label_normalization": True,
        "checkpoint_reused": True,
    }
    (out_dir / "clean_rate_inference_metadata.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )


def predict_protocol(args: argparse.Namespace, root: Path, protocol: str) -> list[dict]:
    if torch is None:
        raise ModuleNotFoundError("torch is required. Run with conda env scellst_bench.")

    from st_pipeline.constants import KEYS
    from st_pipeline.data.collate import mil_collate

    source_tag = args.single_source_tag if protocol == "single" else args.multi_source_tag
    output_tag = args.single_output_tag if protocol == "single" else args.multi_output_tag
    test_slides_arg = args.single_test_slides if protocol == "single" else args.multi_test_slides
    test_slides = mlp_ablation.split_slide_list(test_slides_arg, f"--{protocol}-test-slides")

    ckpt_path = mlp_ablation.require_best_model(root, source_tag)
    kept_genes, gene_ids = load_genes(root, source_tag)
    chunks = core.gene_chunks(len(kept_genes), args.gene_chunk_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out_root = result_dir(root, output_tag)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    model = None
    input_dim = None
    for sid in test_slides:
        slide_out = out_root / sid
        pred_path = slide_out / "pred_rate_bag.npy"
        true_path = slide_out / "true_rate_bag.npy"
        if pred_path.exists() and true_path.exists() and not args.force_predict:
            pred_rate = np.load(pred_path)
            true_rate = np.load(true_path)
            metrics = core.save_metrics(pred_rate, true_rate, kept_genes, slide_out)
            rows.append({"protocol": protocol, "test_slide_id": sid, "result_dir": str(slide_out), **metrics})
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
            model = mlp_ablation.build_model(input_dim, len(kept_genes), args, device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
        assert model is not None

        pred_rate = np.zeros((len(ds), len(kept_genes)), dtype=np.float32)
        true_rate = np.zeros((len(ds), len(kept_genes)), dtype=np.float32)
        offset = 0
        use_amp = bool(args.use_amp and device.type == "cuda")
        with torch.inference_mode():
            for b_idx, batch in enumerate(loader):
                batch = mlp_ablation.move_batch_to_device(batch, device)
                batch_size = int(batch[KEYS.Y_BAG].shape[0])
                sf = batch[KEYS.SIZE_FACTOR].float().clamp_min(1e-8).unsqueeze(1)
                true_rate[offset : offset + batch_size] = (batch[KEYS.Y_BAG].float() / sf).cpu().numpy()

                chunk_preds = []
                for chunk in chunks:
                    idx = torch.as_tensor(chunk, device=device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        mu_rate_bag = predict_rate_chunk(model, batch, idx)
                    chunk_preds.append(mu_rate_bag.float().cpu().numpy())
                pred_rate[offset : offset + batch_size] = np.concatenate(chunk_preds, axis=1)
                offset += batch_size
                if b_idx % args.heartbeat_batches == 0:
                    print(f"[{protocol}:{sid}] clean-rate predict batch {b_idx}/{len(loader)}", flush=True)

        slide_out.mkdir(parents=True, exist_ok=True)
        np.save(pred_path, pred_rate)
        np.save(true_path, true_rate)
        np.save(slide_out / "pred_bag.npy", pred_rate)
        np.save(slide_out / "true_bag.npy", true_rate)
        spot_ids = np.asarray(meta["spot_ids"], dtype=str)[ds.keep_idx]
        (slide_out / "spot_ids.txt").write_text("\n".join(map(str, spot_ids)), encoding="utf-8")
        write_metadata(slide_out, source_tag, output_tag, sid)

        metrics = core.save_metrics(pred_rate, true_rate, kept_genes, slide_out)
        rows.append({"protocol": protocol, "test_slide_id": sid, "result_dir": str(slide_out), **metrics})
        print(f"[{protocol}:{sid}] done in {time.perf_counter() - t0:.1f}s | rate mean={metrics['Mean_Pearson']:.4f}")

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(out_root / "fold_summary_clean_rate.csv", index=False)
    return rows


def main() -> None:
    args = parse_args()
    root = setup_root(args.root)
    protocols = ["single", "multi"] if args.protocol == "both" else [args.protocol]
    rows = []
    for protocol in protocols:
        rows.extend(predict_protocol(args, root, protocol))
    summary = pd.DataFrame(rows)
    out_path = root / "results" / "ablation_decoder" / "mlp_decoder_clean_rate_key_results.csv"
    summary.to_csv(out_path, index=False)
    cols = [
        "protocol",
        "test_slide_id",
        "n_spots",
        "n_genes",
        "Mean_Pearson",
        "Median_Pearson",
        "HVG1000_Mean_Pearson",
        "HVG1000_Median_Pearson",
        "HVG1000_Overlap",
        "Best_Gene",
        "Best_Pearson",
    ]
    print(summary[cols].to_string(index=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
