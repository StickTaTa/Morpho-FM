"""Diagnose the TensorFlow 2.15 / PyTorch 2.4.1+cu118 coexistence problem in `stimage_bench`.

The STimage row must read HEST slides (no torch needed) while using TensorFlow for the
model. `benchmark/hest_dataset.py` imports torch at module level for every other method
in this benchmark, so the STimage path drags torch in behind TensorFlow and the import
fails:

    ImportError: .../torch/lib/libtorch_cuda.so: undefined symbol: ncclCommRegister

The reference project that first adapted STimage to HEST recorded the same class of
problem and resolved it by decoupling the STimage loader from the PyTorch loaders. This
script establishes which import order actually fails, so the fix is chosen from
evidence rather than assumed.

Each mode runs in its own subprocess so a crash cannot affect the others.

Usage:
    python scripts/diagnose_stimage_env.py --out-json results/benchmark_audit/stimage_env_diag.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

MODES = {
    "torch_only": "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)",
    "tensorflow_only": "import tensorflow as tf; print('tf', tf.__version__)",
    "tensorflow_then_torch": (
        "import tensorflow as tf; print('tf', tf.__version__); "
        "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
    ),
    "torch_then_tensorflow": (
        "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); "
        "import tensorflow as tf; print('tf', tf.__version__)"
    ),
    "hest_dataset_import": (
        "import sys; sys.path.insert(0, 'benchmark'); "
        "import hest_dataset; print('hest_dataset ok', hest_dataset.STIMAGE_TILE_SIZE)"
    ),
    "tensorflow_then_hest_dataset": (
        "import tensorflow as tf; print('tf', tf.__version__); "
        "import sys; sys.path.insert(0, 'benchmark'); "
        "import hest_dataset; print('hest_dataset ok', hest_dataset.STIMAGE_TILE_SIZE)"
    ),
    "nccl_probe": (
        "import ctypes, glob, os; "
        "print('LD_LIBRARY_PATH', os.environ.get('LD_LIBRARY_PATH', '')); "
        "print('nccl candidates', glob.glob(os.path.join(os.environ.get('CONDA_PREFIX',''), 'lib', '*nccl*')) "
        "+ glob.glob(os.path.join(os.environ.get('CONDA_PREFIX',''), 'lib/python3.10/site-packages/nvidia/nccl/lib/*')))"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose the stimage_bench import environment.")
    parser.add_argument("--mode", required=True, choices=sorted(MODES))
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--timeout", type=int, default=900)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

    started = __import__("time").time()
    proc = subprocess.run(
        [sys.executable, "-c", MODES[args.mode]],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=int(args.timeout),
    )
    report = {
        "mode": args.mode,
        "returncode": int(proc.returncode),
        "seconds": round(__import__("time").time() - started, 2),
        "python": sys.version.split()[0],
        "conda_prefix": env.get("CONDA_PREFIX", ""),
        "ld_library_path": env.get("LD_LIBRARY_PATH", ""),
        "stdout_tail": proc.stdout.strip().splitlines()[-12:],
        "stderr_tail": proc.stderr.strip().splitlines()[-12:],
        "ok": proc.returncode == 0,
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
