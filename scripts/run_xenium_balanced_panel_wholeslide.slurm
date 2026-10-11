#!/usr/bin/env bash
#SBATCH -p x86_64_gpu800
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -w a800-gpu2
#SBATCH -J xenium_ws29
#SBATCH -o /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/xenium_ws29_%j.log
#SBATCH -e /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/xenium_ws29_%j.log

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

bash scripts/run_118_xenium_balanced_panel_wholeslide_server.sh
