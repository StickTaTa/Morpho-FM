#!/usr/bin/env bash
#SBATCH -p x86_64_gpu6000
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH -c 32
#SBATCH -J h2h3_loso_dense
#SBATCH -o /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/her2st_loso_h2_h3_dense_outputs_a6000_%j.log
#SBATCH -e /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/her2st_loso_h2_h3_dense_outputs_a6000_%j.log

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
GENES=KRT19,ERBB2,KRT14,KRT5,MGP,IGKC

echo "[job] start $(date) on $(hostname)"
nvidia-smi || true

run_target() {
  local target="$1"
  local fold="$2"
  local train_slide="$3"
  local ckpt="$ROOT/checkpoints/her2st_patient_h/st_superres/leave_one_section_out/$fold"

  echo "[${target}] start dense outputs for ${fold} at $(date)"
  test -f "$ckpt/best_model.pt"

  CUDA_VISIBLE_DEVICES=0 python scripts/122_her2st_patient_h_wholeslide.py \
    --root "$ROOT" \
    --train-slide "$train_slide" \
    --section "$target" \
    --tag "$fold" \
    --checkpoint-dir "$ckpt" \
    --out-root "$ROOT/results/her2st_patient_h/whole_slide_loso" \
    --genes "$GENES" \
    --batch-size 8192 \
    --force \
    > "logs/her2st_loso_${target}_dense_gene_${SLURM_JOB_ID}.log" 2>&1 &
  local pid_dense=$!

  CUDA_VISIBLE_DEVICES=1 python scripts/135_her2st_dense_latent_clustering.py \
    --root "$ROOT" \
    --slides "$target" \
    --out-root "$ROOT/results/her2st_patient_h/dense_latent_clustering_loso/$fold" \
    --checkpoint-dir "$ckpt" \
    --n-clusters 10 \
    --batch-size 8192 \
    --sample-size 50000 \
    --kmeans-iters 35 \
    --display-median-size 3 \
    --device cuda \
    --force \
    > "logs/her2st_loso_${target}_dense_cluster_${SLURM_JOB_ID}.log" 2>&1 &
  local pid_cluster=$!

  local status=0
  wait "$pid_dense" || status=1
  wait "$pid_cluster" || status=1
  if [[ "$status" -ne 0 ]]; then
    echo "[${target}] dense outputs failed"
    exit "$status"
  fi

  python scripts/145_her2st_loso_h1_dense_summary.py \
    --root "$ROOT" \
    --fold "$fold" \
    --section "$target" \
    --dense-root "$ROOT/results/her2st_patient_h/whole_slide_loso" \
    --cluster-root "$ROOT/results/her2st_patient_h/dense_latent_clustering_loso" \
    --out-root "$ROOT/results/her2st_patient_h/leave_one_section_out_figures" \
    --genes "$GENES" \
    --dpi 300

  echo "[${target}] finished dense outputs at $(date)"
}

run_target H2 loso_train_H1_H3_to_H2 H1
run_target H3 loso_train_H1_H2_to_H3 H1

echo "[job] finished $(date)"
