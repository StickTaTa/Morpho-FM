#!/usr/bin/env bash
#SBATCH -p x86_64_gpu6000
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH -w a6000-gpu1
#SBATCH -c 32
#SBATCH -J her2st_loso
#SBATCH -o /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/her2st_loso_transfer_a6000_%j.log
#SBATCH -e /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/her2st_loso_transfer_a6000_%j.log

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
  --batch-size 6
  --predict-batch-size 16
  --gene-chunk-size 512
  --num-workers 4
  --epochs 100
  --early-stop-patience 5
  --heartbeat-batches 10
  --force-predict
)

for HOLDOUT in H1 H2 H3; do
  echo "[fold] holdout ${HOLDOUT} start $(date)"
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
    scripts/143_her2st_patient_h_loso_morpho.py \
    --holdout-slide "${HOLDOUT}" \
    "${COMMON_ARGS[@]}" \
    > "logs/her2st_loso_${HOLDOUT}_${SLURM_JOB_ID}.log" 2>&1
  echo "[fold] holdout ${HOLDOUT} finished $(date)"
done

echo "[job] finished $(date)"
