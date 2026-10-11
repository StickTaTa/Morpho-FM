#!/usr/bin/env bash
# Diagnose which import order breaks in `stimage_bench` before choosing a fix.
# All modes run concurrently across the node's four GPUs.
#
# Per the project's remote rules: no --time, no --mem, exclusive node, all GPUs and
# cores requested and threaded.
#
#SBATCH -p x86_64_gpu800
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -c 64
#SBATCH -J stimage_envdiag
#SBATCH -o /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/stimage_env_diag_%j.log
#SBATCH -e /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/stimage_env_diag_%j.log

set -uo pipefail

source /share/home/nxy1zll2hjj/anaconda3/etc/profile.d/conda.sh
conda activate stimage_bench

export OMP_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export MKL_NUM_THREADS=10
export NUMEXPR_NUM_THREADS=10
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd /share/home/nxy1zll2hjj/hjj/Morpho-VC
mkdir -p logs results/benchmark_audit

echo "[node] $(hostname) start=$(date -Is)"
echo "[env] CONDA_PREFIX=$CONDA_PREFIX"
echo "[env] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
python -c "import sys; print('[env] python', sys.version.split()[0])"

MODES=(torch_only tensorflow_only tensorflow_then_torch torch_then_tensorflow hest_dataset_import tensorflow_then_hest_dataset nccl_probe)
pids=()
for i in "${!MODES[@]}"; do
    m="${MODES[$i]}"
    CUDA_VISIBLE_DEVICES="$((i % 4))" python -u scripts/diagnose_stimage_env.py \
        --mode "$m" \
        --out-json "results/benchmark_audit/stimage_env_diag_${m}.json" \
        > "logs/stimage_env_diag_${m}.log" 2>&1 &
    pids+=("$!")
done

for p in "${pids[@]}"; do wait "$p" || true; done

echo "[node] done=$(date -Is)"
for m in "${MODES[@]}"; do
    echo "=============== $m ==============="
    python - "$m" <<'PY'
import json, sys
from pathlib import Path
p = Path(f"results/benchmark_audit/stimage_env_diag_{sys.argv[1]}.json")
if not p.exists():
    print("(no report)")
else:
    r = json.loads(p.read_text())
    print(f"ok={r['ok']} rc={r['returncode']} seconds={r['seconds']}")
    for line in r["stdout_tail"][-3:]:
        print("  out:", line)
    for line in r["stderr_tail"][-3:]:
        print("  err:", line)
PY
done

exit 0
