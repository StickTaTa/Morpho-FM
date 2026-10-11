#!/usr/bin/env bash
#SBATCH -p x86_64_gpu6000
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH -w a6000-gpu2
#SBATCH -c 32
#SBATCH -J h1_loso_dense
#SBATCH -o /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/her2st_loso_h1_dense_outputs_a6000_%j.log
#SBATCH -e /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/her2st_loso_h1_dense_outputs_a6000_%j.log

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
FOLD=loso_train_H2_H3_to_H1
CKPT="$ROOT/checkpoints/her2st_patient_h/st_superres/leave_one_section_out/$FOLD"
GENES=KRT19,ERBB2,KRT14,KRT5,MGP,IGKC

echo "[job] start $(date) on $(hostname)"
nvidia-smi || true

CUDA_VISIBLE_DEVICES=0 python scripts/122_her2st_patient_h_wholeslide.py \
  --root "$ROOT" \
  --train-slide H2 \
  --section H1 \
  --tag "$FOLD" \
  --checkpoint-dir "$CKPT" \
  --out-root "$ROOT/results/her2st_patient_h/whole_slide_loso" \
  --genes "$GENES" \
  --batch-size 8192 \
  --force \
  > "logs/her2st_loso_h1_dense_gene_${SLURM_JOB_ID}.log" 2>&1 &
PID_DENSE=$!

CUDA_VISIBLE_DEVICES=1 python scripts/135_her2st_dense_latent_clustering.py \
  --root "$ROOT" \
  --slides H1 \
  --out-root "$ROOT/results/her2st_patient_h/dense_latent_clustering_loso/$FOLD" \
  --checkpoint-dir "$CKPT" \
  --n-clusters 10 \
  --batch-size 8192 \
  --sample-size 50000 \
  --kmeans-iters 35 \
  --display-median-size 3 \
  --device cuda \
  --force \
  > "logs/her2st_loso_h1_dense_cluster_${SLURM_JOB_ID}.log" 2>&1 &
PID_CLUSTER=$!

wait "$PID_DENSE"
wait "$PID_CLUSTER"

python scripts/145_her2st_loso_h1_dense_summary.py \
  --root "$ROOT" \
  --fold "$FOLD" \
  --section H1 \
  --dense-root "$ROOT/results/her2st_patient_h/whole_slide_loso" \
  --cluster-root "$ROOT/results/her2st_patient_h/dense_latent_clustering_loso" \
  --out-root "$ROOT/results/her2st_patient_h/leave_one_section_out_figures" \
  --genes "$GENES" \
  --dpi 300

echo "[job] finished $(date)"
