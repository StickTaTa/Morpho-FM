#!/usr/bin/env bash
#SBATCH -p x86_64_gpu800
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -w a800-gpu3
#SBATCH -c 64
#SBATCH -J her2st_loso
#SBATCH -o /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/her2st_loso_transfer_a800_%j.log
#SBATCH -e /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/her2st_loso_transfer_a800_%j.log

set -euo pipefail
source /share/home/nxy1zll2hjj/anaconda3/etc/profile.d/conda.sh
conda activate morpho-vc

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export MORPHO_VC_ROOT=/share/home/nxy1zll2hjj/hjj/Morpho-VC
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR=/share/home/nxy1zll2hjj/hjj/Morpho-VC/tmp
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$TMPDIR" logs results/her2st_patient_h/morpho/leave_one_section_out
cd /share/home/nxy1zll2hjj/hjj/Morpho-VC

echo "[job] start $(date) on $(hostname)"
echo "[job] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

COMMON_ARGS=(
  --root /share/home/nxy1zll2hjj/hjj/Morpho-VC
  --mode all
  --batch-size 8
  --predict-batch-size 16
  --gene-chunk-size 1024
  --num-workers 4
  --epochs 100
  --early-stop-patience 5
  --heartbeat-batches 10
  --force-predict
)

run_fold() {
  local holdout="$1"
  local gpus="$2"
  local port="$3"
  echo "[fold] holdout ${holdout} start $(date) on GPUs ${gpus}"
  CUDA_VISIBLE_DEVICES="${gpus}" torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:"${port}" \
    --rdzv_id="her2st_loso_${holdout}_${SLURM_JOB_ID}" \
    scripts/143_her2st_patient_h_loso_morpho.py \
    --holdout-slide "${holdout}" \
    "${COMMON_ARGS[@]}" \
    > "logs/her2st_loso_${holdout}_${SLURM_JOB_ID}.log" 2>&1
  echo "[fold] holdout ${holdout} finished $(date)"
}

run_fold H1 0,1 29531 &
PID1=$!
run_fold H2 2,3 29532 &
PID2=$!

wait "$PID1"
wait "$PID2"

run_fold H3 0,1 29533

echo "[job] finished $(date)"
