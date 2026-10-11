#!/usr/bin/env bash
set -euo pipefail

ROOT="${MORPHO_VC_ROOT:-/share/home/nxy1zll2hjj/hjj/Morpho-VC}"
PYTHON_DEFAULT="/share/home/nxy1zll2hjj/anaconda3/envs/morpho-vc/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_DEFAULT}"

cd "$ROOT"
export MORPHO_VC_ROOT="$ROOT"

echo "=== Host & GPUs ==="
hostname
pwd
date
nvidia-smi -L

echo "=== Python Env ==="
"$PYTHON_BIN" - <<'PY'
import os
import torch
print("python", os.sys.executable)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
print("cwd", os.getcwd())
print("root", os.environ.get("MORPHO_VC_ROOT"))
PY

"$PYTHON_BIN" scripts/118_xenium_balanced_panel_wholeslide.py "$@"
