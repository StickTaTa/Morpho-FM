from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import prostate_protocol_b_morpho as core  # noqa: E402


try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import ConcatDataset, DataLoader, Subset
except (ModuleNotFoundError, ImportError):
    torch = None
    F = None
    nn = None
    ConcatDataset = None
    DataLoader = None
    Subset = None


PROSTATE_SLIDES = ["INT25", "INT26", "INT27", "INT28"]
DEFAULT_SINGLE_TAG = "mlp_decoder_single_INT25_to_INT28"
DEFAULT_MULTI_TAG = "mlp_decoder_multi_INT25_INT26_val_INT27_to_INT28"


def require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("torch is required. Run with conda env scellst_bench.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MLP decoder ablation for prostate Morpho-FM: keep HIPT disk-shaped "
            "MIL bags and the NB objective fixed, but replace the CellFM decoder "
            "with a lightweight morphology-only MLP decoder."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--protocol", choices=["single", "multi", "both"], default="both")
    parser.add_argument("--mode", choices=["train", "predict", "summary", "all"], default="all")
    parser.add_argument("--single-train-slide", default="INT25", choices=PROSTATE_SLIDES)
    parser.add_argument("--single-test-slides", default="INT28")
    parser.add_argument("--multi-train-slides", default="INT25,INT26")
    parser.add_argument("--multi-val-slides", default="INT27")
    parser.add_argument("--multi-test-slides", default="INT28")
    parser.add_argument("--single-tag", default=DEFAULT_SINGLE_TAG)
    parser.add_argument("--multi-tag", default=DEFAULT_MULTI_TAG)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-predict", action="store_true")
    parser.add_argument("--resume-training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--predict-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gene-chunk-size", type=int, default=1024)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-dim", type=int, default=16000)
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


def split_slide_list(value: str, name: str) -> list[str]:
    slides = [x.strip() for x in value.split(",") if x.strip()]
    if not slides:
        raise ValueError(f"{name} must not be empty")
    unknown = sorted(set(slides) - set(PROSTATE_SLIDES))
    if unknown:
        raise ValueError(f"Unknown prostate slide IDs in {name}: {unknown}")
    return slides


def checkpoint_dir(root: Path, tag: str) -> Path:
    return root / "checkpoints" / "st_superres" / "decoder_ablation" / tag


def result_dir(root: Path, tag: str) -> Path:
    return root / "results" / "ablation_decoder" / tag


def best_model_path(root: Path, tag: str) -> Path:
    return checkpoint_dir(root, tag) / "best_model.pt"


def best_model_ready(root: Path, tag: str) -> bool:
    ckpt = best_model_path(root, tag)
    meta = checkpoint_dir(root, tag) / "best_metrics.json"
    return ckpt.exists() and ckpt.stat().st_size > 0 and meta.exists()


def require_best_model(root: Path, tag: str) -> Path:
    ckpt = best_model_path(root, tag)
    if not best_model_ready(root, tag):
        raise FileNotFoundError(
            f"Best model is not ready for tag={tag}. Expected {ckpt} and best_metrics.json."
        )
    return ckpt


class MorphologyMlpMIL(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_genes: int,
        hidden_dim: int = 1536,
        dropout: float = 0.1,
        aggregation: str = "mean",
    ) -> None:
        super().__init__()
        if aggregation != "mean":
            raise ValueError("Only mean aggregation is supported for this ablation.")
        self.n_genes = n_genes
        self.aggregation = aggregation
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.decoder = nn.Linear(hidden_dim, n_genes)
        self.gene_dispersion = nn.Parameter(torch.zeros(n_genes))
        self.softplus = nn.Softplus()

    def _aggregate(self, inst_pred: torch.Tensor, ptr: torch.Tensor, num_bags: int) -> torch.Tensor:
        bag = torch.zeros(num_bags, inst_pred.size(1), device=inst_pred.device, dtype=inst_pred.dtype)
        bag.index_add_(0, ptr, inst_pred)
        counts = torch.bincount(ptr, minlength=num_bags).clamp_min(1).unsqueeze(1).to(inst_pred.dtype)
        return bag / counts

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        gene_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from st_pipeline.constants import KEYS

        x = batch[KEYS.X]
        ptr = batch[KEYS.PTR_BAG_INSTANCE]
        size_factor = batch[KEYS.SIZE_FACTOR]
        h = self.adapter(x)
        if gene_indices is None:
            pred = self.decoder(h)
        else:
            pred = F.linear(h, self.decoder.weight[gene_indices], self.decoder.bias[gene_indices])
        mu_inst = self.softplus(pred)
        num_bags = int(ptr.max().item()) + 1
        mu_bag = self._aggregate(mu_inst, ptr, num_bags)
        mu_bag = mu_bag * size_factor.unsqueeze(1)
        return mu_bag, mu_inst


def build_model(input_dim: int, n_genes: int, args: argparse.Namespace, device: torch.device) -> MorphologyMlpMIL:
    return MorphologyMlpMIL(
        input_dim=input_dim,
        n_genes=n_genes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        aggregation="mean",
    ).to(device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    for key in batch:
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


def iter_loss_chunks(
    model: MorphologyMlpMIL,
    batch: dict,
    chunks: list[list[int]],
    device: torch.device,
    use_amp: bool,
):
    from st_pipeline.constants import KEYS
    from st_pipeline.model.nb_loss import nb_negative_log_likelihood

    for chunk in chunks:
        idx = torch.as_tensor(chunk, device=device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            mu_bag, _ = model(batch, gene_indices=idx)
            theta = torch.exp(model.gene_dispersion[idx])
            loss = nb_negative_log_likelihood(batch[KEYS.Y_BAG][:, idx], mu_bag, theta)
        yield idx, loss


def run_training_loop(
    args: argparse.Namespace,
    root: Path,
    tag: str,
    train_loader,
    val_loader,
    input_dim: int,
    n_genes: int,
    split_payload: dict,
) -> bool:
    require_torch()
    from st_pipeline.constants import KEYS

    ckpt_dir = checkpoint_dir(root, tag)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "split.json").write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = build_model(input_dim, n_genes, args, device)
    resume_path = best_model_path(root, tag)
    if args.resume_training and resume_path.exists():
        model.load_state_dict(torch.load(resume_path, map_location=device))
        print(f"Resumed from {resume_path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    use_amp = bool(args.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    chunks = core.gene_chunks(n_genes, args.gene_chunk_size)
    best_val = 1e9
    no_improve = 0

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tag: {tag}")
    print(f"Device: {device}")
    print(f"Trainable params: {trainable}")
    print(f"Genes: {n_genes} | chunks: {len(chunks)} | gene_chunk_size={args.gene_chunk_size}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for batch_idx, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)
            chunk_order = torch.randperm(len(chunks)).tolist()
            for ci in chunk_order:
                optimizer.zero_grad(set_to_none=True)
                _, loss = next(iter_loss_chunks(model, batch, [chunks[ci]], device, use_amp))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total += float(loss.item())
                steps += 1
            if batch_idx % args.heartbeat_batches == 0:
                print(f"epoch {epoch} batch {batch_idx}/{len(train_loader)}", flush=True)
        train_loss = total / max(steps, 1)

        model.eval()
        total = 0.0
        steps = 0
        with torch.inference_mode():
            for batch in val_loader:
                batch = move_batch_to_device(batch, device)
                for _, loss in iter_loss_chunks(model, batch, chunks, device, use_amp):
                    total += float(loss.item())
                    steps += 1
        val_loss = total / max(steps, 1)
        print(f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}", flush=True)

        if val_loss < best_val - args.early_stop_min_delta:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), resume_path)
            (ckpt_dir / "best_metrics.json").write_text(
                json.dumps({"best_val_loss": best_val, "epoch": epoch}, indent=2),
                encoding="utf-8",
            )
            print(f"Saved best model: {resume_path}")
        else:
            no_improve += 1
            if no_improve >= args.early_stop_patience:
                print(f"Early stop: no improvement for {no_improve} epochs")
                break

    if not best_model_ready(root, tag):
        print(f"[STOP] No best model was saved for tag={tag}; prediction will not run.")
        return False
    return True


def build_single_loaders(args: argparse.Namespace, root: Path, tag: str):
    require_torch()
    from st_pipeline.constants import KEYS
    from st_pipeline.data.collate import mil_collate

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_dir = checkpoint_dir(root, tag)
    kept_genes, gene_ids, _ = core.load_gene_setup(root, ckpt_dir, write_metadata=True)
    base_ds, meta = core.build_slide_dataset(
        root,
        args.single_train_slide,
        kept_genes,
        gene_ids,
        device,
        args.force_cache,
        args.max_dim,
        ensure_cache=True,
    )
    train_indices, val_indices, split_meta = core.make_blocked_split(
        base_ds.locs_px,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    train_ds = Subset(base_ds, train_indices)
    val_ds = Subset(base_ds, val_indices)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=mil_collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=mil_collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    sample_batch = next(iter(train_loader))
    input_dim = int(sample_batch[KEYS.X].shape[-1])
    split_payload = {
        "protocol": "mlp_decoder_ablation_single_blocked",
        "train_slide_id": args.single_train_slide,
        "test_slide_ids": split_slide_list(args.single_test_slides, "--single-test-slides"),
        "validation_split": split_meta,
        "train_spots": len(train_ds),
        "val_spots": len(val_ds),
        "radius_px": float(meta["radius_px"]),
        "mask_size": int(meta["mask_size"]),
    }
    return train_loader, val_loader, input_dim, len(kept_genes), split_payload


def build_multi_loaders(args: argparse.Namespace, root: Path, tag: str):
    require_torch()
    from st_pipeline.constants import KEYS
    from st_pipeline.data.collate import mil_collate

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_dir = checkpoint_dir(root, tag)
    kept_genes, gene_ids, _ = core.load_gene_setup(root, ckpt_dir, write_metadata=True)
    train_slides = split_slide_list(args.multi_train_slides, "--multi-train-slides")
    val_slides = split_slide_list(args.multi_val_slides, "--multi-val-slides")
    test_slides = split_slide_list(args.multi_test_slides, "--multi-test-slides")
    overlap = (set(train_slides) & set(val_slides)) | (set(train_slides) & set(test_slides)) | (set(val_slides) & set(test_slides))
    if overlap:
        raise ValueError(f"Train, validation and test slides must be disjoint: {sorted(overlap)}")

    datasets = {}
    metas = {}
    for sid in list(dict.fromkeys(train_slides + val_slides)):
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
        datasets[sid] = ds
        metas[sid] = meta
    train_ds = ConcatDataset([datasets[sid] for sid in train_slides])
    val_ds = ConcatDataset([datasets[sid] for sid in val_slides])
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=mil_collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=mil_collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    sample_batch = next(iter(train_loader))
    input_dim = int(sample_batch[KEYS.X].shape[-1])
    split_payload = {
        "protocol": "mlp_decoder_ablation_multi",
        "train_slide_ids": train_slides,
        "val_slide_ids": val_slides,
        "test_slide_ids": test_slides,
        "train_spots_per_slide": {sid: len(datasets[sid]) for sid in train_slides},
        "val_spots_per_slide": {sid: len(datasets[sid]) for sid in val_slides},
        "radius_px": {sid: float(metas[sid]["radius_px"]) for sid in metas},
        "mask_size": {sid: int(metas[sid]["mask_size"]) for sid in metas},
    }
    return train_loader, val_loader, input_dim, len(kept_genes), split_payload


def train_protocol(args: argparse.Namespace, root: Path, protocol: str, tag: str) -> bool:
    seed_everything(args.seed)
    if protocol == "single":
        train_loader, val_loader, input_dim, n_genes, split_payload = build_single_loaders(args, root, tag)
    elif protocol == "multi":
        train_loader, val_loader, input_dim, n_genes, split_payload = build_multi_loaders(args, root, tag)
    else:
        raise ValueError(protocol)
    return run_training_loop(args, root, tag, train_loader, val_loader, input_dim, n_genes, split_payload)


def predict_protocol(args: argparse.Namespace, root: Path, protocol: str, tag: str) -> None:
    require_torch()
    from st_pipeline.constants import KEYS
    from st_pipeline.data.collate import mil_collate
    from st_pipeline.data.gene_vocab import load_gene_vocab, map_genes_to_vocab

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    ckpt_path = require_best_model(root, tag)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ckpt_dir = checkpoint_dir(root, tag)
    out_root = result_dir(root, tag)
    out_root.mkdir(parents=True, exist_ok=True)
    kept_genes = json.loads((ckpt_dir / "common_genes.json").read_text(encoding="utf-8"))
    vocab = load_gene_vocab(root / "assets" / "cellfm" / "expand_gene_info.csv")
    kept_genes, gene_ids = map_genes_to_vocab(kept_genes, vocab)
    chunks = core.gene_chunks(len(kept_genes), args.gene_chunk_size)

    if protocol == "single":
        test_slides = split_slide_list(args.single_test_slides, "--single-test-slides")
    else:
        test_slides = split_slide_list(args.multi_test_slides, "--multi-test-slides")

    rows = []
    model = None
    input_dim = None
    for sid in test_slides:
        slide_out = out_root / sid
        pred_path = slide_out / "pred_bag.npy"
        true_path = slide_out / "true_bag.npy"
        if pred_path.exists() and true_path.exists() and not args.force_predict:
            print(f"[{sid}] cached predictions found.")
            pred = np.load(pred_path)
            true = np.load(true_path)
            metrics = core.save_metrics(pred, true, kept_genes, slide_out)
            rows.append({"test_slide_id": sid, **metrics})
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
            model = build_model(input_dim, len(kept_genes), args, device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
        assert model is not None

        pred = np.zeros((len(ds), len(kept_genes)), dtype=np.float32)
        true = np.zeros((len(ds), len(kept_genes)), dtype=np.float32)
        offset = 0
        iterator = tqdm(loader, total=len(loader), desc=f"{sid} predict", ncols=100) if tqdm else loader
        use_amp = bool(args.use_amp and device.type == "cuda")
        with torch.inference_mode():
            for b_idx, batch in enumerate(iterator):
                batch = move_batch_to_device(batch, device)
                batch_size = batch[KEYS.Y_BAG].shape[0]
                true[offset : offset + batch_size] = batch[KEYS.Y_BAG].cpu().numpy()
                chunk_preds = []
                for chunk in chunks:
                    idx = torch.as_tensor(chunk, device=device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        mu_bag, _ = model(batch, gene_indices=idx)
                    chunk_preds.append(mu_bag.float().cpu().numpy())
                pred[offset : offset + batch_size] = np.concatenate(chunk_preds, axis=1)
                offset += batch_size
                if tqdm is None and b_idx % args.heartbeat_batches == 0:
                    print(f"[{sid}] predict batch {b_idx}/{len(loader)}", flush=True)

        slide_out.mkdir(parents=True, exist_ok=True)
        np.save(pred_path, pred)
        np.save(true_path, true)
        spot_ids = np.asarray(meta["spot_ids"], dtype=str)[ds.keep_idx]
        (slide_out / "spot_ids.txt").write_text("\n".join(map(str, spot_ids)), encoding="utf-8")
        metrics = core.save_metrics(pred, true, kept_genes, slide_out)
        rows.append({"test_slide_id": sid, **metrics})
        print(f"[{sid}] done in {time.perf_counter() - t0:.1f}s | mean={metrics['Mean_Pearson']:.4f}")

    core.write_fold_summary(rows, out_root)


def summarize_protocol(args: argparse.Namespace, root: Path, protocol: str, tag: str) -> None:
    require_best_model(root, tag)
    if protocol == "single":
        train_desc = args.single_train_slide
        val_desc = "blocked"
        test_slides = split_slide_list(args.single_test_slides, "--single-test-slides")
    else:
        train_desc = ",".join(split_slide_list(args.multi_train_slides, "--multi-train-slides"))
        val_desc = ",".join(split_slide_list(args.multi_val_slides, "--multi-val-slides"))
        test_slides = split_slide_list(args.multi_test_slides, "--multi-test-slides")

    out_root = result_dir(root, tag)
    rows = []
    missing = []
    for sid in test_slides:
        metrics_path = out_root / sid / "metrics.json"
        pred_path = out_root / sid / "pred_bag.npy"
        true_path = out_root / sid / "true_bag.npy"
        if not metrics_path.exists() or not pred_path.exists() or not true_path.exists():
            missing.append(sid)
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "method": "HIPT + MLP decoder",
                "protocol": protocol,
                "train_slides": train_desc,
                "val_slides": val_desc,
                "test_slide_id": sid,
                "status": "complete",
                "result_dir": str(out_root / sid),
                **metrics,
            }
        )
    if missing:
        raise FileNotFoundError(f"Prediction outputs are incomplete for tag={tag}: missing {missing}")
    if not rows:
        raise RuntimeError(f"No completed predictions found for tag={tag}")
    df = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / "fold_status.csv", index=False)
    df.to_csv(out_root / "decoder_ablation_summary.csv", index=False)
    print(f"Saved summary: {out_root / 'decoder_ablation_summary.csv'}")
    print(df)


def run_protocol(args: argparse.Namespace, root: Path, protocol: str, tag: str) -> None:
    should_continue = True
    if args.mode in {"train", "all"}:
        should_continue = train_protocol(args, root, protocol, tag)
    if args.mode in {"predict", "all"}:
        if not should_continue or not best_model_ready(root, tag):
            raise FileNotFoundError(f"Training did not produce a best model for {tag}; skipping prediction.")
        predict_protocol(args, root, protocol, tag)
    if args.mode in {"summary", "all", "predict"}:
        summarize_protocol(args, root, protocol, tag)


def main() -> None:
    args = parse_args()
    root = setup_root(args.root)
    require_torch()
    protocols = ["single", "multi"] if args.protocol == "both" else [args.protocol]
    for protocol in protocols:
        tag = args.single_tag if protocol == "single" else args.multi_tag
        print(f"=== MLP decoder ablation: {protocol} ({tag}) ===", flush=True)
        run_protocol(args, root, protocol, tag)


if __name__ == "__main__":
    main()
