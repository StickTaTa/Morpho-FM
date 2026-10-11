from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


PROSTATE_SLIDES = ["INT25", "INT26", "INT27", "INT28"]
FOLD_TABLE = {
    "INT25": {"val": "INT26", "train": ["INT27", "INT28"]},
    "INT26": {"val": "INT27", "train": ["INT25", "INT28"]},
    "INT27": {"val": "INT28", "train": ["INT25", "INT26"]},
    "INT28": {"val": "INT25", "train": ["INT26", "INT27"]},
}
PROTOCOL = "prostate_protocol_a_rotating_multi_slide"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate prostate Protocol-A Morpho-FM fold status and metrics.")
    parser.add_argument("--root", type=Path, default=None)
    return parser.parse_args()


def setup_root(root_arg: Path | None) -> Path:
    root = root_arg or Path(os.environ.get("MORPHO_VC_ROOT", "."))
    return root.expanduser().resolve()


def expected_folds() -> list[dict]:
    folds = []
    for test_slide in PROSTATE_SLIDES:
        spec = FOLD_TABLE[test_slide]
        tag = f"test_{test_slide}_val_{spec['val']}"
        folds.append(
            {
                "protocol": PROTOCOL,
                "tag": tag,
                "train_slides": spec["train"],
                "val_slide": spec["val"],
                "test_slide_id": test_slide,
            }
        )
    return folds


def checkpoint_dir(root: Path, tag: str) -> Path:
    return root / "checkpoints" / "st_superres" / "protocol_a_rotating" / f"cellfm_multi_{tag}"


def result_dir(root: Path, tag: str) -> Path:
    return root / "results" / "prostate_hest" / "morpho" / "multi_rotating" / tag


def row_status(ckpt_path: Path, slide_dir: Path, metrics_path: Path) -> str:
    if metrics_path.exists():
        return "complete"
    if not (ckpt_path / "best_model.pt").exists():
        return "missing_checkpoint"
    if not (slide_dir / "pred_bag.npy").exists() or not (slide_dir / "true_bag.npy").exists():
        return "missing_prediction"
    return "missing_metrics"


def build_rows(root: Path) -> list[dict]:
    rows = []
    for fold in expected_folds():
        ckpt_path = checkpoint_dir(root, fold["tag"])
        out_root = result_dir(root, fold["tag"])
        slide_dir = out_root / fold["test_slide_id"]
        metrics_path = slide_dir / "metrics.json"
        split_path = ckpt_path / "split.json"
        split_meta = {}
        if split_path.exists():
            split = json.loads(split_path.read_text(encoding="utf-8"))
            split_meta = {
                "split_train_slides": ",".join(split.get("train_ids", [])),
                "split_val_slide": ",".join(split.get("val_ids", [])),
                "validation_strategy": split.get("validation_strategy"),
            }
        row = {
            "method": "Morpho-FM",
            "protocol": PROTOCOL,
            "tag": fold["tag"],
            "train_slides": ",".join(fold["train_slides"]),
            "val_slide": fold["val_slide"],
            "test_slide_id": fold["test_slide_id"],
            "status": row_status(ckpt_path, slide_dir, metrics_path),
            "checkpoint_dir": str(ckpt_path),
            "result_dir": str(slide_dir),
            **split_meta,
        }
        if metrics_path.exists():
            row.update(json.loads(metrics_path.read_text(encoding="utf-8")))
        rows.append(row)
    return rows


def main() -> None:
    root = setup_root(parse_args().root)
    summary_root = root / "results" / "prostate_hest" / "morpho" / "protocol_a_rotating"
    summary_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(build_rows(root))
    status_path = summary_root / "prostate_protocol_a_morpho_status.csv"
    complete_path = summary_root / "prostate_protocol_a_morpho_complete_metrics.csv"
    df.to_csv(status_path, index=False)
    df[df["status"].eq("complete")].to_csv(complete_path, index=False)
    print(f"Saved status table: {status_path}")
    print(f"Saved complete metrics table: {complete_path}")
    print(df[["tag", "train_slides", "val_slide", "test_slide_id", "status"]])


if __name__ == "__main__":
    main()
