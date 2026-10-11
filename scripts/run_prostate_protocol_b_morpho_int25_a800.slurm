#!/usr/bin/env bash
#SBATCH -p x86_64_gpu800
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -J prostate_pb_INT25_A800
#SBATCH -o /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/prostate_pb_INT25_A800_%j.log
#SBATCH -e /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/prostate_pb_INT25_A800_%j.log

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

# The old prostate folds used 256 genes per chunk. A800 has substantially more
# memory headroom, so start with 1024 to reduce chunk-loop overhead. If this
# fails early with CUDA OOM, resubmit with GENE_CHUNK_SIZE=512.
export GENE_CHUNK_SIZE="${GENE_CHUNK_SIZE:-1024}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-8}"

cd /share/home/nxy1zll2hjj/hjj/Morpho-VC
mkdir -p logs

bash scripts/run_prostate_protocol_b_morpho_fold.sh INT25
