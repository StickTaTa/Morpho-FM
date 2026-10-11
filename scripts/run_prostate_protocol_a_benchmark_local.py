from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "results" / "benchmark_audit"
MANIFEST_PATH = AUDIT_DIR / "prostate_protocol_a_benchmark_manifest.csv"
GENERATED_DIR = AUDIT_DIR / "protocol_a_run_scripts"
LOG_DIR = AUDIT_DIR / "protocol_a_run_logs"
STATUS_PATH = AUDIT_DIR / "prostate_protocol_a_benchmark_run_status.csv"
EVENTS_PATH = AUDIT_DIR / "prostate_protocol_a_benchmark_run_events.jsonl"

FOLDS = [
    {"tag": "test_INT25_val_INT26", "train_slides": ["INT27", "INT28"], "val_slide": "INT26", "test_slide": "INT25"},
    {"tag": "test_INT26_val_INT27", "train_slides": ["INT25", "INT28"], "val_slide": "INT27", "test_slide": "INT26"},
    {"tag": "test_INT27_val_INT28", "train_slides": ["INT25", "INT26"], "val_slide": "INT28", "test_slide": "INT27"},
    {"tag": "test_INT28_val_INT25", "train_slides": ["INT26", "INT27"], "val_slide": "INT25", "test_slide": "INT28"},
]

METHODS = {
    "HisToGene": {
        "method_dir": "histogene",
        "template": ROOT / "benchmark" / "01_HisToGene_benchmark.ipynb",
        "env": "histogene_bench",
        "cells": [1, 3, 5, 7, 9, 11],
    },
    "iStar": {
        "method_dir": "istar",
        "template": ROOT / "benchmark" / "02_iStar_benchmark.ipynb",
        "env": "istar_bench",
        "cells": [1, 2, 4, 6, 8, 10, 12],
    },
    "mclSTExp": {
        "method_dir": "mclstexp",
        "template": ROOT / "benchmark" / "03_mclSTExp_benchmark.ipynb",
        "env": "scellst_bench",
        "cells": [1, 2, 4, 6, 8, 9, 10],
    },
    "sCellST": {
        "method_dir": "scellst",
        "template": ROOT / "benchmark" / "04_sCellST_benchmark.ipynb",
        "env": "scellst_bench",
        "cells": [1, 3, 5, 7],
    },
    "HiST": {
        "method_dir": "hist",
        "template": ROOT / "benchmark" / "06_HiST_benchmark.ipynb",
        "env": "scellst_bench",
        "cells": [1, 2, 3, 4, 5, 6],
    },
    "Path2Space": {
        "method_dir": "path2space",
        "template": ROOT / "benchmark" / "07_Path2Space_benchmark.ipynb",
        "env": "scellst_bench",
        "cells": [1, 2, 3, 4, 5, 6],
    },
    "ST-Net": {
        "method_dir": "stnet",
        "template": ROOT / "benchmark" / "09_STNet_benchmark.ipynb",
        "env": "scellst_bench",
        "cells": [1, 2, 3, 4, 5, 6],
    },
    # Registered so the fold tasks reach the Protocol-A manifest through the same path
    # as every other row, but not runnable by this script's local `conda run`: STimage
    # needs GPU TensorFlow and there is no GPU TensorFlow for Windows. Execution goes
    # through the remote Slurm path; see benchmark/STimage/README_MORPHOFM_ADAPTATION.md.
    "STimage": {
        "method_dir": "stimage",
        "template": ROOT / "benchmark" / "08_STimage_benchmark.ipynb",
        "env": "stimage_bench",
        "cells": [1, 2, 3, 4, 5, 6],
    },
}


def env_python(env_name: str) -> str:
    candidate = Path("C:/ProgramData/anaconda3/envs") / env_name / "python.exe"
    if candidate.exists():
        return str(candidate)
    return "python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run missing prostate Protocol-A multi-slide baseline benchmark folds sequentially.")
    parser.add_argument("--methods", nargs="*", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--fold-tags", nargs="*", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
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
    fieldnames = [
        "time",
        "method",
        "fold_tag",
        "train_slides",
        "val_slide",
        "test_slide",
        "env",
        "status",
        "returncode",
        "seconds",
        "script_path",
        "log_path",
    ]
    with STATUS_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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


def result_dir_for(method_dir: str, fold: dict) -> Path:
    return ROOT / "results" / "prostate_hest" / method_dir / "multi_rotating" / fold["tag"] / fold["test_slide"]


def py_list(items: list[str]) -> str:
    return repr(list(items))


def patch_common(code: str, method_dir: str, fold: dict) -> str:
    out_dir = result_dir_for(method_dir, fold)
    train_ids = py_list(fold["train_slides"])
    val_ids = py_list([fold["val_slide"]])
    test_ids = py_list([fold["test_slide"]])
    code = code.replace("train_ids = ['INT25', 'INT26']", f"train_ids = {train_ids}")
    code = code.replace('train_ids = ["INT25", "INT26"]', f"train_ids = {train_ids}")
    code = code.replace("val_ids = ['INT27']", f"val_ids = {val_ids}")
    code = code.replace('val_ids = ["INT27"]', f"val_ids = {val_ids}")
    code = code.replace("test_ids = ['INT28']", f"test_ids = {test_ids}")
    code = code.replace('test_ids = ["INT28"]', f"test_ids = {test_ids}")
    code = code.replace("target_slide = 'INT28'", f"target_slide = '{fold['test_slide']}'")
    code = code.replace("train_slides = ['INT25', 'INT26']", f"train_slides = {train_ids}")
    code = code.replace("val_slides = ['INT27']", f"val_slides = {val_ids}")
    code = code.replace("TEST_ID = \"INT28\"", f"TEST_ID = \"{fold['test_slide']}\"")
    code = code.replace("workers = 4", "workers = 0")
    return code


def patch_result_roots(code: str, method: str, method_dir: str, fold: dict) -> str:
    out = result_dir_for(method_dir, fold)
    out_expr = f"ROOT / 'results' / 'prostate_hest' / '{method_dir}' / 'multi_rotating' / '{fold['tag']}' / '{fold['test_slide']}'"
    if method == "HisToGene":
        code = code.replace("result_dir = ROOT / 'benchmark' / 'results' / 'histogene'", f"result_dir = {out_expr}")
    elif method == "iStar":
        code = code.replace("RESULT_DIR = ROOT / 'benchmark' / 'results' / 'istar'", f"RESULT_DIR = {out_expr}")
        code = code.replace("DATA_ROOT = RESULT_DIR / 'data' # iStar 脚本通常在 data/ 目录下找数据", "DATA_ROOT = ROOT / 'benchmark' / 'results' / 'istar' / 'data_protocol_a'")
        code = code.replace("DATA_ROOT = RESULT_DIR / 'data'", "DATA_ROOT = ROOT / 'benchmark' / 'results' / 'istar' / 'data_protocol_a'")
    elif method == "mclSTExp":
        code = code.replace("RESULT_DIR = ROOT / 'benchmark' / 'results' / 'mclSTExp'", f"RESULT_DIR = {out_expr}")
        code = code.replace("DATA_ROOT = RESULT_DIR / 'data'", "DATA_ROOT = ROOT / 'benchmark' / 'results' / 'mclSTExp' / 'data_protocol_a'")
    elif method == "sCellST":
        code = code.replace("RESULT_DIR = BENCHMARK_DIR / 'results' / 'sCellST'", f"RESULT_DIR = {out_expr}")
        code = code.replace("config.data.list_training_ids = train_ids + val_ids", "config.data.list_training_ids = train_ids")
    elif method == "HiST":
        code = code.replace('result_dir = ROOT / "benchmark" / "results" / "hist"', f"result_dir = {out_expr}")
    elif method == "Path2Space":
        code = code.replace('result_dir = ROOT / "benchmark" / "results" / "path2space"', f"result_dir = {out_expr}")
    elif method == "ST-Net":
        code = code.replace('result_dir = ROOT / "benchmark" / "results" / "stnet"', f"result_dir = {out_expr}")
    elif method == "STimage":
        code = code.replace('result_dir = ROOT / "benchmark" / "results" / "stimage"', f"result_dir = {out_expr}")
    if str(out) in code:
        pass
    return code


def patch_istar(code: str, fold: dict) -> str:
    injected = (
        f"print(f\"Common Genes: {{len(common_genes)}}\")\n"
        f"train_ids = {py_list(fold['train_slides'])}\n"
        f"val_ids = {py_list([fold['val_slide']])}\n"
        f"test_ids = {py_list([fold['test_slide']])}\n"
        f"print('Protocol-A split:', train_ids, val_ids, test_ids)\n"
    )
    code = code.replace("print(f\"Common Genes: {len(common_genes)}\")", injected)
    old_block = """# 训练
if not (DATA_ROOT / 'INT26' / 'states').exists():
    last_trained_id = train_istar_seq(['INT25', 'INT26'], epochs=200)
else:
    print(\"Training seems done. Using INT26 states.\")
    last_trained_id = 'INT26'
"""
    new_block = """# 训练: always rebuild fold-specific states, but reuse converted/preprocessed slide cache.
for _sid in train_ids + test_ids:
    shutil.rmtree(DATA_ROOT / _sid / 'states', ignore_errors=True)
    shutil.rmtree(DATA_ROOT / _sid / 'cnts-super', ignore_errors=True)
last_trained_id = train_istar_seq(train_ids, epochs=200)
"""
    code = code.replace(old_block, new_block)
    code = code.replace("predict_istar(last_trained_id, 'INT28')", "predict_istar(last_trained_id, test_ids[0])")
    code = code.replace("pred_bag, true_bag, genes = load_istar_predictions('INT28')", "pred_bag, true_bag, genes = load_istar_predictions(test_ids[0])")
    return code


def postlude(method: str, method_dir: str, fold: dict) -> str:
    cleanup = ""
    if method == "iStar":
        cleanup = """
if os.environ.get("BENCHMARK_CLEAN_ISTAR_CACHE", "1") == "1":
    for _sid in train_ids + test_ids:
        shutil.rmtree(DATA_ROOT / _sid / "states", ignore_errors=True)
        shutil.rmtree(DATA_ROOT / _sid / "cnts-super", ignore_errors=True)
"""
    elif method == "mclSTExp":
        cleanup = """
if os.environ.get("BENCHMARK_CLEAN_MCL_CHUNKS", "1") == "1":
    for _p in RESULT_DIR.glob("preds_chunk_*.npy"):
        _p.unlink(missing_ok=True)
"""
    return f"""

# %% protocol-A postlude
import json as _json
from pathlib import Path as _Path
# Prefer the notebook's own `result_dir` when it defines one: that path was rewritten by
# `patch_result_roots` to the directory the run actually writes to, whereas the literal
# below is a path on the machine that generated this script. Using the literal on a
# different machine (the STimage row runs on the cluster, not here) creates a directory
# named after the local Windows path instead of writing where the run expects.
if "result_dir" in globals():
    _out_dir = _Path(result_dir)
else:
    _out_dir = _Path({str(result_dir_for(method_dir, fold))!r})
_out_dir.mkdir(parents=True, exist_ok=True)
if "pred_bag" not in globals() and "full_preds" in globals():
    pred_bag = full_preds
if "true_bag" not in globals() and "ground_truth" in globals():
    true_bag = ground_truth
if "RESULT_DIR" not in globals():
    RESULT_DIR = _out_dir
if "result_dir" not in globals():
    result_dir = _out_dir
if "pred_bag" in globals() and "true_bag" in globals():
    np.save(_out_dir / "pred_bag.npy", pred_bag)
    np.save(_out_dir / "true_bag.npy", true_bag)
_genes = None
for _name in ["common_genes", "genes", "gene_names"]:
    if _name in globals():
        _genes = globals()[_name]
        break
if _genes is not None:
    pd.Series([x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in list(_genes)]).to_csv(
        _out_dir / "gene_names.txt", index=False, header=False
    )
(_out_dir / "protocol_config.json").write_text(
    _json.dumps(
        {{
            "protocol": "rotating_heldout_multi_slide",
            "method": {method!r},
            "fold_tag": {fold["tag"]!r},
            "train_slides": {fold["train_slides"]!r},
            "val_slide": {fold["val_slide"]!r},
            "test_slide": {fold["test_slide"]!r},
        }},
        indent=2,
    ),
    encoding="utf-8",
)
{cleanup}
print("Protocol-A outputs verified at", _out_dir)
"""


def patch_code(method: str, method_dir: str, code: str, fold: dict) -> str:
    if method == "HisToGene":
        code = code.replace("import scanpy as sc\n", "")
        code = code.replace("from utils import *\n", "")
        code = code.replace(
            'else:\n    print("No existing checkpoint found. Proceeding to training.")\n\nif train_model:',
            'else:\n    print("No existing checkpoint found. Proceeding to training.")\n    train_model = True\n\nif train_model:',
        )
    code = patch_common(code, method_dir, fold)
    code = patch_result_roots(code, method, method_dir, fold)
    if method == "iStar":
        code = patch_istar(code, fold)
    return code + postlude(method, method_dir, fold)


def is_complete(method_dir: str, fold: dict) -> bool:
    result_dir = result_dir_for(method_dir, fold)
    return all((result_dir / name).exists() for name in ["pred_bag.npy", "true_bag.npy", "metrics.json"])


def task_groups(args: argparse.Namespace) -> list[dict]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}. Run build_prostate_protocol_a_benchmark_manifest.py first.")
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = manifest[manifest["method"].isin(args.methods)].copy()
    if args.fold_tags:
        manifest = manifest[manifest["fold_tag"].isin(args.fold_tags)]
    if not args.force:
        manifest = manifest[manifest["status"].ne("complete")]
    tasks = []
    for method in args.methods:
        for fold in FOLDS:
            if args.fold_tags and fold["tag"] not in set(args.fold_tags):
                continue
            rows = manifest[(manifest["method"] == method) & (manifest["fold_tag"] == fold["tag"])]
            if rows.empty and not args.force:
                continue
            meta = METHODS[method]
            tasks.append({"method": method, "method_dir": meta["method_dir"], "fold_tag": fold["tag"], **fold})
    return tasks


def generated_script_path(method: str, fold: dict) -> Path:
    return GENERATED_DIR / method / f"{fold['tag']}.py"


def log_path(method: str, fold: dict) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / method / f"{fold['tag']}_{stamp}.log"


def archive_mcl_checkpoints(result_dir: Path) -> None:
    checkpoints = result_dir / "checkpoints"
    if not checkpoints.exists():
        return
    archive_root = Path("D:/Morpho-FM_archive/mclSTExp_protocol_a_checkpoints")
    rel = result_dir.relative_to(ROOT)
    target_parent = archive_root / rel
    target_parent.mkdir(parents=True, exist_ok=True)
    target = target_parent / "checkpoints"
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(checkpoints), str(target_parent))


def run_task(task: dict) -> int:
    method = task["method"]
    method_meta = METHODS[method]
    code = load_notebook_code(method_meta["template"], method_meta["cells"])
    code = patch_code(method, method_meta["method_dir"], code, task)
    script_path = generated_script_path(method, task)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Auto-generated by scripts/run_prostate_protocol_a_benchmark_local.py\n"
        f"# method={method} fold={task['fold_tag']} train={','.join(task['train_slides'])} val={task['val_slide']} test={task['test_slide']}\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
    )
    script_path.write_text(header + "\n" + code, encoding="utf-8")

    lp = log_path(method, task)
    lp.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MORPHO_VC_ROOT"] = str(ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"
    env.setdefault("BENCHMARK_CLEAN_ISTAR_CACHE", "1")
    env.setdefault("BENCHMARK_CLEAN_MCL_CHUNKS", "1")
    activate_bat = ROOT.parent / "ProgramData" / "anaconda3" / "Scripts" / "activate.bat"
    if not activate_bat.exists():
        activate_bat = Path(r"C:\ProgramData\anaconda3\Scripts\activate.bat")
    cmd = f'cmd.exe /d /c call "{activate_bat}" {method_meta["env"]} && python -u "{script_path}"'
    write_event({"event": "start", **task, "env": method_meta["env"], "script_path": str(script_path), "log_path": str(lp)})
    start = time.time()
    with lp.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"[start] {now()}\n")
        log.write(f"[cmd] {cmd}\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        seconds = time.time() - start
        log.write(f"\n[end] {now()} returncode={proc.returncode} seconds={seconds:.1f}\n")

    result_dir = result_dir_for(method_meta["method_dir"], task)
    if proc.returncode == 0 and method == "mclSTExp":
        archive_mcl_checkpoints(result_dir)
    status = "success" if proc.returncode == 0 and is_complete(method_meta["method_dir"], task) else "failed"
    row = {
        "method": method,
        "fold_tag": task["fold_tag"],
        "train_slides": ",".join(task["train_slides"]),
        "val_slide": task["val_slide"],
        "test_slide": task["test_slide"],
        "env": method_meta["env"],
        "status": status,
        "returncode": proc.returncode,
        "seconds": f"{seconds:.1f}",
        "script_path": str(script_path),
        "log_path": str(lp),
    }
    append_status(row)
    write_event({"event": "end", **row})
    return 0 if status == "success" else 1


def main() -> None:
    args = parse_args()
    tasks = task_groups(args)
    if args.max_runs is not None:
        tasks = tasks[: args.max_runs]
    print(f"[tasks] {len(tasks)}")
    for task in tasks:
        print(f" - {task['method']} {','.join(task['train_slides'])} | val {task['val_slide']} -> {task['test_slide']}")
    if args.dry_run:
        return
    failed = 0
    for task in tasks:
        rc = run_task(task)
        if rc != 0:
            failed += 1
            print(f"[WARN] {task['method']} {task['fold_tag']} failed; continuing to next task.")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build_prostate_protocol_a_benchmark_manifest.py")], cwd=str(ROOT))
    print(f"[done] requested tasks completed; failed={failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
