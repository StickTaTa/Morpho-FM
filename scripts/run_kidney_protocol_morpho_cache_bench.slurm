#!/usr/bin/env bash
#SBATCH -p x86_64_gpu6000
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH -J kidney_pb_probe
#SBATCH -o /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/kidney_pb_probe_%j.log
#SBATCH -e /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/kidney_pb_probe_%j.log

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

python scripts/kidney_protocol_morpho.py \
  --protocol multi_loso \
  --test-slide INT21 \
  --mode cache

python scripts/kidney_protocol_morpho.py \
  --protocol single_rotating \
  --train-slide INT18 \
  --mode bench \
  --bench-batch-sizes 4,8,16 \
  --bench-gene-chunk-sizes 256,512 \
  --probe-batches 2

python scripts/kidney_protocol_morpho.py \
  --protocol multi_loso \
  --test-slide INT21 \
  --mode bench \
  --bench-batch-sizes 4,8,16 \
  --bench-gene-chunk-sizes 256,512 \
  --probe-batches 2
