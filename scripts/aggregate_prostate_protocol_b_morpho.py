from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


PROSTATE_SLIDES = ["INT25", "INT26", "INT27", "INT28"]
PROTOCOL = "prostate_protocol_b_rotating_single_slide"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Morpho-FM prostate protocol-B rotating single-slide fold status and metrics."
    )
    parser.add_argument("--root", type=Path, default=None, help="Project root. Defaults to MORPHO_VC_ROOT or cwd.")
    return parser.parse_args()


def setup_root(root_arg: Path | None) -> Path:
    root = root_arg or Path(os.environ.get("MORPHO_VC_ROOT", "."))
    return root.expanduser().resolve()


def fold_tag(train_slide: str, test_slides: list[str]) -> str:
    return f"{train_slide}_to_{'_'.join(test_slides)}"


def checkpoint_dir(root: Path, tag: str) -> Path:
    return root / "checkpoints" / "st_superres" / "protocol_b_rotating" / f"cellfm_single_{tag}"


def result_dir(root: Path, tag: str) -> Path:
    return root / "results" / "prostate_hest" / "morpho" / "single_superres" / tag


def expected_folds() -> list[tuple[str, list[str], str]]:
    folds = []
    for train_slide in PROSTATE_SLIDES:
        test_slides = [sid for sid in PROSTATE_SLIDES if sid != train_slide]
        folds.append((train_slide, test_slides, fold_tag(train_slide, test_slides)))
    return folds


def status_for_paths(ckpt_path: Path, slide_dir: Path, metrics_path: Path) -> str:
    if metrics_path.exists():
        return "complete"
    if not (ckpt_path / "best_model.pt").exists():
        return "missing_checkpoint"
    if not (slide_dir / "pred_bag.npy").exists() or not (slide_dir / "true_bag.npy").exists():
        return "missing_prediction"
    return "missing_metrics"


def build_rows(root: Path) -> list[dict]:
    rows = []
    for train_slide, test_slides, tag in expected_folds():
        ckpt_path = checkpoint_dir(root, tag)
        out_root = result_dir(root, tag)
        split_path = ckpt_path / "split.json"
        fold_meta = {}
        if split_path.exists():
            split = json.loads(split_path.read_text(encoding="utf-8"))
            fold_meta = {
                "validation_strategy": split.get("validation_split", {}).get("strategy"),
                "validation_grid_size": split.get("validation_split", {}).get("grid_size"),
                "validation_actual_ratio": split.get("validation_split", {}).get("val_ratio_actual"),
            }
        for test_slide in test_slides:
            slide_dir = out_root / test_slide
            metrics_path = slide_dir / "metrics.json"
            status = status_for_paths(ckpt_path, slide_dir, metrics_path)
            row = {
                "method": "Morpho-FM",
                "protocol": PROTOCOL,
                "train_slide": train_slide,
                "test_slide_id": test_slide,
                "fold_tag": tag,
                "status": status,
                "checkpoint_dir": str(ckpt_path),
                "result_dir": str(slide_dir),
                **fold_meta,
            }
            if metrics_path.exists():
                row.update(json.loads(metrics_path.read_text(encoding="utf-8")))
            rows.append(row)
    return rows


def main() -> None:
    root = setup_root(parse_args().root)
    summary_root = root / "results" / "prostate_hest" / "protocol_b_rotating"
    summary_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(build_rows(root))
    status_path = summary_root / "prostate_protocol_b_morpho_status.csv"
    complete_path = summary_root / "prostate_protocol_b_morpho_complete_metrics.csv"
    df.to_csv(status_path, index=False)
    df[df["status"].eq("complete")].to_csv(complete_path, index=False)

    print(f"Saved status table: {status_path}")
    print(f"Saved complete metrics table: {complete_path}")
    print(df[["train_slide", "test_slide_id", "fold_tag", "status"]])


if __name__ == "__main__":
    main()
