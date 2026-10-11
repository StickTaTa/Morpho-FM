#!/usr/bin/env bash
#SBATCH -p x86_64_gpu800
#SBATCH --nodelist=a800-gpu2
#SBATCH --gres=gpu:2
#SBATCH -c 32
#SBATCH -J prostate_A_INT27_A800x2
#SBATCH -o /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/prostate_protocol_a_INT27_a800x2_%j.log
#SBATCH -e /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/prostate_protocol_a_INT27_a800x2_%j.log

set -euo pipefail
source /share/home/nxy1zll2hjj/anaconda3/etc/profile.d/conda.sh
conda activate morpho-vc

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MORPHO_VC_ROOT=/share/home/nxy1zll2hjj/hjj/Morpho-VC
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PROSTATE_A_BATCH_SIZE="${PROSTATE_A_BATCH_SIZE:-4}"
export PROSTATE_A_GENE_CHUNK_SIZE="${PROSTATE_A_GENE_CHUNK_SIZE:-898}"
export PROSTATE_A_PREDICT_BATCH_SIZE="${PROSTATE_A_PREDICT_BATCH_SIZE:-8}"
export PROSTATE_A_NPROC="${PROSTATE_A_NPROC:-2}"

cd "$MORPHO_VC_ROOT"
mkdir -p logs run_locks

bash scripts/run_prostate_protocol_a_morpho_fold.sh INT27
