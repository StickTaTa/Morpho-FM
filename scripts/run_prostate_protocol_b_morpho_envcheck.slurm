#!/usr/bin/env bash
#SBATCH -p x86_64_gpu800
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -J prostate_pb_env
#SBATCH -o /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/prostate_pb_env_%j.log
#SBATCH -e /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/prostate_pb_env_%j.log

set -euo pipefail
source /share/home/nxy1zll2hjj/anaconda3/etc/profile.d/conda.sh
conda activate morpho-vc

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export MORPHO_VC_ROOT=/share/home/nxy1zll2hjj/hjj/Morpho-VC
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd /share/home/nxy1zll2hjj/hjj/Morpho-VC
mkdir -p logs

hostname
date
nvidia-smi -L
python scripts/prostate_protocol_b_morpho.py --help >/tmp/prostate_protocol_b_morpho_help.txt
python - <<'PY'
import os
import sys
import torch
sys.path.append("/share/home/nxy1zll2hjj/hjj/Morpho-VC/src")
from st_pipeline.superres.feature_cache import get_common_genes
print("python", sys.executable)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("common_genes", len(get_common_genes(["INT25", "INT26", "INT27", "INT28"])))
print("root", os.environ.get("MORPHO_VC_ROOT"))
PY
