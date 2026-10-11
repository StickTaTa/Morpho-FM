from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "results" / "benchmark_audit"
MANIFEST_PATH = AUDIT_DIR / "benchmark_task_manifest.csv"
GENERATED_DIR = AUDIT_DIR / "run_scripts"
LOG_DIR = AUDIT_DIR / "run_logs"
STATUS_PATH = AUDIT_DIR / "benchmark_sequential_run_status.csv"
EVENTS_PATH = AUDIT_DIR / "benchmark_sequential_run_events.jsonl"

PROSTATE_SLIDES = ["INT25", "INT26", "INT27", "INT28"]
KIDNEY_SLIDES = ["INT13", "INT14", "INT15", "INT17", "INT18", "INT19", "INT21", "INT24"]

METHODS = {
    "HisToGene": {
        "method_dir": "histogene",
        "template": ROOT / "benchmark" / "101_HisToGene_benchmark_single_INT28.ipynb",
        "env": "histogene_bench",
        "cells": [1, 3, 5, 7, 9, 14],
    },
    "iStar": {
        "method_dir": "istar",
        "template": ROOT / "benchmark" / "102_iStar_benchmark_single_INT28.ipynb",
        "env": "istar_bench",
        "cells": [1, 3, 5, 7, 9, 11, 15],
    },
    "mclSTExp": {
        "method_dir": "mclstexp",
        "template": ROOT / "benchmark" / "103_mclSTExp_benchmark_single_INT28.ipynb",
        "env": "scellst_bench",
        "cells": [1, 3, 5, 7, 9, 11, 15],
    },
    "sCellST": {
        "method_dir": "scellst",
        "template": ROOT / "benchmark" / "104_sCellST_benchmark_single_INT28.ipynb",
        "env": "scellst_bench",
        "cells": [1, 3, 5, 7, 9, 12],
    },
    "HiST": {
        "method_dir": "hist",
        "template": ROOT / "benchmark" / "106_HiST_benchmark_single_INT28.ipynb",
        "env": "scellst_bench",
        "cells": [1, 2, 3, 4, 5, 6, 9],
    },
    "Path2Space": {
        "method_dir": "path2space",
        "template": ROOT / "benchmark" / "107_Path2Space_benchmark_single_INT28.ipynb",
        "env": "scellst_bench",
        "cells": [1, 2, 3, 4, 5, 6, 9],
    },
    "ST-Net": {
        "method_dir": "stnet",
        "template": ROOT / "benchmark" / "109_STNet_benchmark_single_INT28.ipynb",
        "env": "scellst_bench",
        "cells": [1, 2, 3, 4, 5, 6, 9],
    },
    # STimage is registered here so its tasks reach the manifests and the metric
    # pipeline through the same path as every other row, but it cannot run through
    # this script's local `conda run`: it needs GPU TensorFlow, and Windows wheels for
    # TensorFlow are CPU-only from 2.11 on. Execution goes through
    # `scripts/run_stimage_remote.py`, which reuses this module's cell extraction and
    # patching and submits to Slurm instead.
    "STimage": {
        "method_dir": "stimage",
        "template": ROOT / "benchmark" / "108_STimage_benchmark_single_INT28.ipynb",
        "env": "stimage_bench",
        "cells": [1, 2, 3, 4, 5, 6, 9],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run missing rotating single-slide benchmark folds sequentially.")
    parser.add_argument("--cohort", choices=["Prostate", "Kidney"], default="Prostate")
    parser.add_argument("--protocol", choices=["single", "multi"], default="single")
    parser.add_argument("--methods", nargs="*", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--train-slides", nargs="*", default=None)
    parser.add_argument("--fold-tags", nargs="*", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Run even if all expected outputs are already complete.")
    return parser.parse_args()


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_event(payload: dict) -> None:
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": now(), **payload}, ensure_ascii=False) + "\n")


def append_status(row: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = STATUS_PATH.exists()
    with STATUS_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time",
                "cohort",
                "method",
                "train_slide",
                "test_slides",
                "fold_tag",
                "env",
                "status",
                "returncode",
                "seconds",
                "script_path",
                "log_path",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow({"time": now(), **row})


def load_notebook_code(template: Path, cell_indices: list[int]) -> str:
    nb = json.loads(template.read_text(encoding="utf-8"))
    chunks = []
    for idx in cell_indices:
        cell = nb["cells"][idx]
        if cell["cell_type"] != "code":
            raise ValueError(f"Cell {idx} in {template.name} is not code.")
        chunks.append(f"\n# %% notebook cell {idx}\n")
        chunks.append("".join(cell.get("source", "")))
        chunks.append("\n")
    return "\n".join(chunks)


def patch_code(code: str, cohort_dir: str, train_slide: str, test_slides: list[str]) -> str:
    first_test = test_slides[0]
    fold = f"{train_slide}_to_{'_'.join(test_slides)}"
    code = code.replace("prostate_hest", cohort_dir)
    if cohort_dir == "kidney_hest":
        kidney_istar_cache = "ROOT / 'benchmark' / 'results' / 'kidney_INT21' / 'istar' / 'data'"
        code = code.replace(
            "DATA_CACHE_DIR = ROOT / 'benchmark' / 'results' / 'istar' / 'data' # Re-using preprocessed data",
            f"DATA_CACHE_DIR = {kidney_istar_cache} # Re-using preprocessed kidney iStar data",
        )
        code = code.replace(
            "DATA_CACHE_DIR = ROOT / 'benchmark' / 'results' / 'istar' / 'data'",
            f"DATA_CACHE_DIR = {kidney_istar_cache}",
        )
    code = code.replace("INT25_to_INT26_INT27_INT28", fold)
    code = code.replace("INT25_to_INT28", f"{train_slide}_to_{first_test}")
    code = code.replace("TRAIN_SLIDE_ID = 'INT25'", f"TRAIN_SLIDE_ID = '{train_slide}'")
    code = code.replace('TRAIN_SLIDE_ID = "INT25"', f'TRAIN_SLIDE_ID = "{train_slide}"')
    code = code.replace("TEST_SLIDE_ID = 'INT28'", f"TEST_SLIDE_ID = '{first_test}'")
    code = code.replace('TEST_SLIDE_ID = "INT28"', f'TEST_SLIDE_ID = "{first_test}"')
    code = code.replace(
        'SUPPLEMENT_TEST_SLIDE_IDS = ["INT26", "INT27", "INT28"]',
        f"SUPPLEMENT_TEST_SLIDE_IDS = {json.dumps(test_slides)}",
    )
    code = code.replace("NUM_WORKERS = 4 if HAS_OPENSLIDE else 0", "NUM_WORKERS = 0")
    return code


def generated_script_path(cohort: str, method: str, train_slide: str) -> Path:
    return GENERATED_DIR / cohort.lower() / method / f"{train_slide}.py"


def log_path(cohort: str, method: str, train_slide: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / cohort.lower() / method / f"{train_slide}_{stamp}.log"


def expected_result_dirs(cohort_dir: str, method_dir: str, train_slide: str, test_slides: list[str]) -> list[Path]:
    fold = f"{train_slide}_to_{'_'.join(test_slides)}"
    return [ROOT / "results" / cohort_dir / method_dir / "single" / fold / sid for sid in test_slides]


def is_complete(cohort_dir: str, method_dir: str, train_slide: str, test_slides: list[str]) -> bool:
    for d in expected_result_dirs(cohort_dir, method_dir, train_slide, test_slides):
        if not ((d / "metrics.json").exists() and (d / "pred_bag.npy").exists() and (d / "true_bag.npy").exists()):
            return False
    return True


def task_groups(args: argparse.Namespace) -> list[dict]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}. Run build_benchmark_task_manifest.py first.")
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = manifest[manifest["cohort"].eq(args.cohort) & manifest["method"].isin(args.methods)].copy()
    if args.train_slides:
        manifest = manifest[manifest["train_slide"].isin(args.train_slides)]
    manifest = manifest[manifest["status"].ne("complete")]
    groups_by_key = {}
    for (cohort, cohort_dir, method, method_dir, train_slide, fold_tag), df in manifest.groupby(
        ["cohort", "cohort_dir", "method", "method_dir", "train_slide", "fold_tag"],
        sort=True,
    ):
        test_slides = sorted(df["test_slide"].astype(str).tolist())
        groups_by_key[(method, train_slide)] = {
            "cohort": cohort,
            "cohort_dir": cohort_dir,
            "method": method,
            "method_dir": method_dir,
            "train_slide": train_slide,
            "test_slides": test_slides,
            "fold_tag": fold_tag,
        }
    slide_order = PROSTATE_SLIDES if args.cohort == "Prostate" else KIDNEY_SLIDES
    groups = []
    for method in args.methods:
        for train_slide in slide_order:
            task = groups_by_key.get((method, train_slide))
            if task is not None:
                groups.append(task)
    return groups


def run_task(task: dict) -> int:
    method_meta = METHODS[task["method"]]
    code = load_notebook_code(method_meta["template"], method_meta["cells"])
    code = patch_code(code, task["cohort_dir"], task["train_slide"], task["test_slides"])
    script_path = generated_script_path(task["cohort"], task["method"], task["train_slide"])
    script_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Auto-generated by scripts/run_benchmark_missing_sequential.py\n"
        f"# cohort={task['cohort']} method={task['method']} train={task['train_slide']} tests={','.join(task['test_slides'])}\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
    )
    script_path.write_text(header + "\n" + code, encoding="utf-8")

    lp = log_path(task["cohort"], task["method"], task["train_slide"])
    lp.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MORPHO_VC_ROOT"] = str(ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLBACKEND"] = "Agg"
    # Without this, stdout is block-buffered when redirected to the log file and a
    # long training run produces no visible progress until the buffer fills.
    env["PYTHONUNBUFFERED"] = "1"
    cmd = ["conda", "run", "--no-capture-output", "-n", method_meta["env"], "python", str(script_path)]
    write_event({"event": "start", **task, "env": method_meta["env"], "script_path": str(script_path), "log_path": str(lp)})
    start = time.time()
    with lp.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"[start] {now()}\n")
        log.write(f"[cmd] {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        seconds = time.time() - start
        log.write(f"\n[end] {now()} returncode={proc.returncode} seconds={seconds:.1f}\n")
    status = "success" if proc.returncode == 0 and is_complete(task["cohort_dir"], task["method_dir"], task["train_slide"], task["test_slides"]) else "failed"
    row = {
        "cohort": task["cohort"],
        "method": task["method"],
        "train_slide": task["train_slide"],
        "test_slides": ",".join(task["test_slides"]),
        "fold_tag": task["fold_tag"],
        "env": method_meta["env"],
        "status": status,
        "returncode": proc.returncode,
        "seconds": f"{seconds:.1f}",
        "script_path": str(script_path),
        "log_path": str(lp),
    }
    append_status(row)
    write_event({"event": "end", **row})
    return proc.returncode


def main() -> None:
    args = parse_args()
    if args.protocol == "multi":
        if args.cohort != "Prostate":
            raise ValueError("The multi-slide benchmark path is currently defined for prostate Protocol A only.")
        cmd = [sys.executable, str(ROOT / "scripts" / "build_prostate_protocol_a_benchmark_manifest.py")]
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_prostate_protocol_a_benchmark_local.py"),
            "--methods",
            *args.methods,
        ]
        if args.fold_tags:
            cmd += ["--fold-tags", *args.fold_tags]
        if args.max_runs is not None:
            cmd += ["--max-runs", str(args.max_runs)]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.force:
            cmd.append("--force")
        raise SystemExit(subprocess.run(cmd, cwd=str(ROOT)).returncode)

    groups = task_groups(args)
    if args.max_runs is not None:
        groups = groups[: args.max_runs]
    print(f"[tasks] {len(groups)} groups")
    for task in groups:
        print(f" - {task['cohort']} {task['method']} {task['train_slide']} -> {','.join(task['test_slides'])}")
    if args.dry_run:
        return
    for i, task in enumerate(groups, 1):
        meta = METHODS[task["method"]]
        if not args.force and is_complete(task["cohort_dir"], task["method_dir"], task["train_slide"], task["test_slides"]):
            print(f"[{i}/{len(groups)}] skip complete {task['method']} {task['train_slide']}")
            continue
        print(f"[{i}/{len(groups)}] running {task['cohort']} {task['method']} {task['train_slide']} in {meta['env']}")
        rc = run_task(task)
        if rc != 0:
            print(f"[WARN] task failed with returncode={rc}; continuing to next task.")
        # Refresh the manifest so manual outputs or just-finished runs are visible to later skips.
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build_benchmark_task_manifest.py")], cwd=str(ROOT))


if __name__ == "__main__":
    main()
