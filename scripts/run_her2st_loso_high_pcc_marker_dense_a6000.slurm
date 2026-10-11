#!/usr/bin/env bash
#SBATCH -p x86_64_gpu6000
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH -c 32
#SBATCH -J her2st_loso_markers
#SBATCH -o /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/her2st_loso_high_pcc_marker_dense_a6000_%j.log
#SBATCH -e /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/her2st_loso_high_pcc_marker_dense_a6000_%j.log

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
mkdir -p "$TMPDIR" logs

cd /share/home/nxy1zll2hjj/hjj/Morpho-VC

ROOT=/share/home/nxy1zll2hjj/hjj/Morpho-VC
OUT_ROOT="$ROOT/results/her2st_patient_h/whole_slide_loso_high_pcc_markers"
GENES=ERBB2,KRT19,CD24,MGP,CD74,IGKC

echo "[job] start $(date) on $(hostname)"
nvidia-smi || true

run_dense() {
  local target="$1"
  local fold="$2"
  local train_slide="$3"
  local gpu="$4"
  local ckpt="$ROOT/checkpoints/her2st_patient_h/st_superres/leave_one_section_out/$fold"

  echo "[${target}] dense marker prediction for ${fold} on GPU ${gpu} at $(date)"
  test -f "$ckpt/best_model.pt"

  CUDA_VISIBLE_DEVICES="$gpu" python scripts/122_her2st_patient_h_wholeslide.py \
    --root "$ROOT" \
    --train-slide "$train_slide" \
    --section "$target" \
    --tag "$fold" \
    --checkpoint-dir "$ckpt" \
    --out-root "$OUT_ROOT" \
    --genes "$GENES" \
    --batch-size 8192 \
    --force \
    > "logs/her2st_loso_${target}_high_pcc_marker_dense_${SLURM_JOB_ID}.log" 2>&1

  echo "[${target}] done at $(date)"
}

run_dense H1 loso_train_H2_H3_to_H1 H2 0 &
PID_H1=$!
run_dense H2 loso_train_H1_H3_to_H2 H1 1 &
PID_H2=$!
wait "$PID_H1"
wait "$PID_H2"

run_dense H3 loso_train_H1_H2_to_H3 H1 0

echo "[job] finished $(date)"
