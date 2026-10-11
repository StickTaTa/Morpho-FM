#!/usr/bin/env bash
#SBATCH -p x86_64_gpu800
#SBATCH --nodelist=a800-gpu1
#SBATCH --gres=gpu:3
#SBATCH -J prostate_pb_INT25_A800x3
#SBATCH -o /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/prostate_pb_INT25_A800x3_%j.log
#SBATCH -e /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/prostate_pb_INT25_A800x3_%j.log

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

# 1024 reaches A800 memory saturation and OOMs on this fold. 896 is the next
# high-throughput setting to keep the cards heavily used while leaving headroom.
export GENE_CHUNK_SIZE="${GENE_CHUNK_SIZE:-896}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd /share/home/nxy1zll2hjj/hjj/Morpho-VC
mkdir -p logs

bash scripts/run_prostate_protocol_b_morpho_fold.sh INT25
