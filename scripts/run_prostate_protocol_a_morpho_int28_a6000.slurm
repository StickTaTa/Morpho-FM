#!/usr/bin/env bash
#SBATCH -p x86_64_gpu6000
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH -J prostate_A_INT28
#SBATCH -o /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/prostate_protocol_a_INT28_a6000_%j.log
#SBATCH -e /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/prostate_protocol_a_INT28_a6000_%j.log

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
export PROSTATE_A_BATCH_SIZE="${PROSTATE_A_BATCH_SIZE:-2}"
export PROSTATE_A_GENE_CHUNK_SIZE="${PROSTATE_A_GENE_CHUNK_SIZE:-512}"
export PROSTATE_A_PREDICT_BATCH_SIZE="${PROSTATE_A_PREDICT_BATCH_SIZE:-8}"
export PROSTATE_A_NPROC="${PROSTATE_A_NPROC:-2}"

cd "$MORPHO_VC_ROOT"
mkdir -p logs run_locks

bash scripts/run_prostate_protocol_a_morpho_fold.sh INT28
