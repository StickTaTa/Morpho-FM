#!/usr/bin/env bash
#SBATCH -p x86_64_gpu800
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -J prostate_pb_morpho
#SBATCH --array=0-2
#SBATCH -o /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/prostate_pb_morpho_%A_%a.log
#SBATCH -e /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/prostate_pb_morpho_%A_%a.log

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

TRAIN_SLIDES=(INT26 INT27 INT28)
TRAIN_SLIDE="${TRAIN_SLIDES[$SLURM_ARRAY_TASK_ID]}"

bash scripts/run_prostate_protocol_b_morpho_fold.sh "$TRAIN_SLIDE"
