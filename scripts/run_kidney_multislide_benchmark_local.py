from __future__ import annotations

"""Run the two kidney benchmark settings that the 2026-09-23 extension never built.

Background
----------
Path2Space, STimage and ST-Net were added to the benchmark on 2026-09-23 with only the
prostate templates (`0X_*_benchmark.ipynb` and `10X_*_single_INT28.ipynb`). The 01-06
families each also carry a kidney multi-slide template (`0X_*_benchmark_kidney_INT21.ipynb`)
and a kidney single-slide template (`10X_*_single_kidney_INT18_to_INT21.ipynb`), and those
were never created for 07/08/09. As a result the three methods have no kidney multi-slide
row and no kidney INT18-to-INT21 single-slide row -- both of which Fig. 3a needs.

The six missing templates now exist (see the generator that produced them). This script
executes them.

Settings
--------
multi   train INT13/INT14/INT15/INT17/INT18/INT19, val INT24, test INT21
        output -> benchmark/results/kidney_INT21/<method_dir>/
        (the same directory the five 2026-04 baselines write to, so Fig. 3a picks it up)
single  train INT18, test INT21
        output -> results/kidney_hest/<method_dir>/single/INT18_to_INT21/

STimage is registered here so it appears in the same logs, but it is not runnable by this
script: it needs GPU TensorFlow and there is no GPU TensorFlow for Windows. It goes
through the remote Slurm path instead.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "results" / "benchmark_audit"
STATUS_PATH = AUDIT_DIR / "kidney_extension_run_status.csv"
EVENTS_PATH = AUDIT_DIR / "kidney_extension_run_events.jsonl"

METHODS = {
    "Path2Space": {
        "method_dir": "path2space",
        "env": "scellst_bench",
        "multi_template": ROOT / "benchmark" / "07_Path2Space_benchmark_kidney_INT21.ipynb",
        "single_template": ROOT / "benchmark" / "107_Path2Space_benchmark_single_kidney_INT18_to_INT21.ipynb",
    },
    "ST-Net": {
        "method_dir": "stnet",
        "env": "scellst_bench",
        "multi_template": ROOT / "benchmark" / "09_STNet_benchmark_kidney_INT21.ipynb",
        "single_template": ROOT / "benchmark" / "109_STNet_benchmark_single_kidney_INT18_to_INT21.ipynb",
    },
    # registered for the record; runs on the cluster, not here
    "STimage": {
        "method_dir": "stimage",
        "env": "stimage_bench",
        "multi_template": ROOT / "benchmark" / "08_STimage_benchmark_kidney_INT21.ipynb",
        "single_template": ROOT / "benchmark" / "108_STimage_benchmark_single_kidney_INT18_to_INT21.ipynb",
        "remote_only": True,
    },
}

# cells 1..6 carry the run; cell 7 is the top-4-gene visualisation the other runners skip
CELLS = [1, 2, 3, 4, 5, 6]

PROTOCOLS = {
    "multi": {
        "result_dir": lambda md: ROOT / "benchmark" / "results" / "kidney_INT21" / md,
        "expected": lambda md: [ROOT / "benchmark" / "results" / "kidney_INT21" / md],
    },
    "single": {
        "result_dir": lambda md: ROOT / "results" / "kidney_hest" / md / "single" / "INT18_to_INT21",
        "expected": lambda md: [ROOT / "results" / "kidney_hest" / md / "single" / "INT18_to_INT21"],
    },
}


def env_python(env_name: str) -> str:
    candidate = Path("C:/ProgramData/anaconda3/envs") / env_name / "python.exe"
    return str(candidate) if candidate.exists() else "python"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_status(row: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = STATUS_PATH.exists()
    fields = ["time", "method", "protocol", "env", "status", "returncode", "seconds",
              "result_dir", "script_path", "log_path"]
    with STATUS_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({"time": now(), **row})


def write_event(payload: dict) -> None:
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": now(), **payload}, ensure_ascii=False) + "\n")


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


def is_complete(result_dir: Path) -> bool:
    return all((result_dir / name).exists() for name in ("metrics.json", "pred_bag.npy", "true_bag.npy"))


def summarise(result_dir: Path) -> str:
    try:
        m = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - only on a broken run
        return f"<metrics unreadable: {exc}>"
    return "Mean_Pearson=%.6f MAE=%.4f RMSE=%.4f best=%s(%.3f)" % (
        m.get("Mean_Pearson", float("nan")), m.get("MAE", float("nan")), m.get("RMSE", float("nan")),
        m.get("Best_Gene", "?"), m.get("Best_Pearson") or float("nan"))


def run_one(method: str, protocol: str, dry_run: bool, force: bool) -> dict:
    meta = METHODS[method]
    if meta.get("remote_only"):
        print("  [skip] %s %s runs on the cluster (GPU TensorFlow), not here" % (method, protocol))
        return {"method": method, "protocol": protocol, "status": "remote"}

    template = meta["multi_template"] if protocol == "multi" else meta["single_template"]
    method_dir = meta["method_dir"]
    result_dir = PROTOCOLS[protocol]["result_dir"](method_dir)

    if not template.exists():
        raise FileNotFoundError(template)
    if not force and is_complete(result_dir):
        print("  [skip] %s %s already complete: %s" % (method, protocol, summarise(result_dir)))
        return {"method": method, "protocol": protocol, "status": "already_complete"}

    code = load_notebook_code(template, CELLS)
    script_path = AUDIT_DIR / "kidney_extension_run_scripts" / protocol / f"{method}.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(
        "# Auto-generated by scripts/run_kidney_multislide_benchmark_local.py\n"
        f"# method={method} protocol={protocol} template={template.name}\n"
        "import matplotlib\nmatplotlib.use('Agg')\n" + code,
        encoding="utf-8")

    log_path = AUDIT_DIR / "kidney_extension_run_logs" / protocol / (
        f"{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [env_python(meta["env"]), "-u", str(script_path)]
    print("  [run ] %s %s  env=%s  ->  %s" % (method, protocol, meta["env"], result_dir))
    print("         %s" % " ".join(cmd))
    if dry_run:
        return {"method": method, "protocol": protocol, "status": "dry_run", "result_dir": str(result_dir)}

    env = os.environ.copy()
    env["MORPHO_VC_ROOT"] = str(ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"

    write_event({"event": "start", "method": method, "protocol": protocol,
                 "script_path": str(script_path), "log_path": str(log_path)})
    start = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"[start] {now()}\n[cmd] {' '.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        seconds = time.time() - start
        log.write(f"\n[end] {now()} returncode={proc.returncode} seconds={seconds:.1f}\n")

    ok = proc.returncode == 0 and is_complete(result_dir)
    status = "success" if ok else "failed"
    row = {"method": method, "protocol": protocol, "env": meta["env"], "status": status,
           "returncode": proc.returncode, "seconds": f"{seconds:.1f}", "result_dir": str(result_dir),
           "script_path": str(script_path), "log_path": str(log_path)}
    append_status(row)
    write_event({"event": "end", **row})
    print("  [done] %s %s  %s  (%.1f min)" % (method, protocol, status, seconds / 60))
    if ok:
        print("         %s" % summarise(result_dir))
    else:
        print("         see %s" % log_path)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the missing kidney benchmark settings for the three added methods.")
    ap.add_argument("--methods", nargs="*", default=["Path2Space", "ST-Net"], choices=list(METHODS))
    ap.add_argument("--protocols", nargs="*", default=["multi", "single"], choices=list(PROTOCOLS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    print("kidney extension: %d method(s) x %d protocol(s)" % (len(args.methods), len(args.protocols)))
    for method in args.methods:
        for protocol in args.protocols:
            run_one(method, protocol, args.dry_run, args.force)


if __name__ == "__main__":
    main()
