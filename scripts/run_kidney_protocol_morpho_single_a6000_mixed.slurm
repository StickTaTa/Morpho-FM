#!/usr/bin/env bash
#SBATCH -p x86_64_gpu6000
#SBATCH --gres=gpu:2
#SBATCH -c 30
#SBATCH -J kidney_single_mx
#SBATCH --array=1-6
#SBATCH -o /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/kidney_single_mixed_%A_%a.log
#SBATCH -e /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/kidney_single_mixed_%A_%a.log

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
export KIDNEY_BATCH_SIZE=4
export KIDNEY_GENE_CHUNK_SIZE=512
export KIDNEY_PREDICT_BATCH_SIZE=8
export KIDNEY_NPROC=2

cd /share/home/nxy1zll2hjj/hjj/Morpho-VC
mkdir -p logs

TRAIN_SLIDES=(INT13 INT14 INT15 INT17 INT18 INT19 INT21 INT24)
TRAIN_SLIDE="${TRAIN_SLIDES[$SLURM_ARRAY_TASK_ID]}"

bash scripts/run_kidney_protocol_morpho_fold.sh single_rotating "$TRAIN_SLIDE"
