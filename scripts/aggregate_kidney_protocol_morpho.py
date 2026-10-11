from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


KIDNEY_SLIDES = ["INT13", "INT14", "INT15", "INT17", "INT18", "INT19", "INT21", "INT24"]
SINGLE_TRAIN_SLIDES = KIDNEY_SLIDES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate kidney Morpho-FM rotating protocol status.")
    parser.add_argument("--root", type=Path, default=None)
    return parser.parse_args()


def setup_root(root_arg: Path | None) -> Path:
    root = root_arg or Path(os.environ.get("MORPHO_VC_ROOT", "."))
    return root.expanduser().resolve()


def cyclic_val_slide(test_slide: str) -> str:
    idx = KIDNEY_SLIDES.index(test_slide)
    return KIDNEY_SLIDES[(idx + 1) % len(KIDNEY_SLIDES)]


def expected_folds() -> list[dict]:
    rows = []
    for train_slide in SINGLE_TRAIN_SLIDES:
        test_slides = [sid for sid in KIDNEY_SLIDES if sid != train_slide]
        tag = f"train_{train_slide}_to_{'_'.join(test_slides)}"
        rows.append(
            {
                "protocol": "single_rotating",
                "tag": tag,
                "train_ids": [train_slide],
                "val_ids": [],
                "test_slides": test_slides,
            }
        )
    return rows


def checkpoint_dir(root: Path, protocol: str, tag: str) -> Path:
    return root / "checkpoints" / "kidney_hest" / "st_superres" / "protocol_rotating" / protocol / tag


def result_dir(root: Path, protocol: str, tag: str) -> Path:
    return root / "results" / "kidney_hest" / "morpho" / protocol / tag


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
        ckpt_path = checkpoint_dir(root, fold["protocol"], fold["tag"])
        out_root = result_dir(root, fold["protocol"], fold["tag"])
        split_path = ckpt_path / "split.json"
        split_meta = {}
        if split_path.exists():
            split = json.loads(split_path.read_text(encoding="utf-8"))
            split_meta = {
                "split_train_ids": ",".join(split.get("train_ids", [])),
                "split_val_ids": ",".join(split.get("val_ids", [])),
                "validation_strategy": split.get("validation_strategy"),
            }
        for test_slide in fold["test_slides"]:
            slide_dir = out_root / test_slide
            metrics_path = slide_dir / "metrics.json"
            row = {
                "method": "Morpho-FM",
                "protocol": fold["protocol"],
                "tag": fold["tag"],
                "train_ids": ",".join(fold["train_ids"]),
                "val_ids": ",".join(fold["val_ids"]),
                "test_slide_id": test_slide,
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
    summary_root = root / "results" / "kidney_hest" / "morpho" / "protocol_rotating"
    summary_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(build_rows(root))
    status_path = summary_root / "kidney_protocol_morpho_status.csv"
    complete_path = summary_root / "kidney_protocol_morpho_complete_metrics.csv"
    df.to_csv(status_path, index=False)
    df[df["status"].eq("complete")].to_csv(complete_path, index=False)
    print(f"Saved status table: {status_path}")
    print(f"Saved complete metrics table: {complete_path}")
    print(df[["protocol", "tag", "test_slide_id", "status"]])


if __name__ == "__main__":
    main()
