#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/huangjinjin/Desktop/Morpho-VC}"
CONDA_BIN="${CONDA_BIN:-/home/huangjinjin/miniconda3/bin/activate}"

TRAIN_SLIDES=(${TRAIN_SLIDES:-H1 H2 H3})
BATCH_SIZE="${BATCH_SIZE:-6}"
PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-16}"
GENE_CHUNK_SIZE="${GENE_CHUNK_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-100}"
HEARTBEAT_BATCHES="${HEARTBEAT_BATCHES:-10}"
WHOLESLIDE_TOP_N="${WHOLESLIDE_TOP_N:-24}"
WHOLESLIDE_BATCH_SIZE="${WHOLESLIDE_BATCH_SIZE:-8192}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

source "$CONDA_BIN" morpho-vc
cd "$ROOT"

echo "[start] HER2ST patient H pipeline"
echo "ROOT=$ROOT"
echo "TRAIN_SLIDES=${TRAIN_SLIDES[*]}"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "PREDICT_BATCH_SIZE=$PREDICT_BATCH_SIZE"
echo "GENE_CHUNK_SIZE=$GENE_CHUNK_SIZE"
echo "NUM_WORKERS=$NUM_WORKERS"
echo "EPOCHS=$EPOCHS"
echo "WHOLESLIDE_TOP_N=$WHOLESLIDE_TOP_N"
echo "WHOLESLIDE_BATCH_SIZE=$WHOLESLIDE_BATCH_SIZE"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "---GPU---"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv,noheader

echo "[step] cache H1/H2/H3"
CUDA_VISIBLE_DEVICES=0 python scripts/121_her2st_patient_h_morpho.py \
  --root "$ROOT" \
  --train-slide H1 \
  --test-slides H1,H2,H3 \
  --mode cache \
  --num-workers "$NUM_WORKERS" \
  --max-dim 16000

for sid in "${TRAIN_SLIDES[@]}"; do
  echo "[step] measured clustering $sid"
  CUDA_VISIBLE_DEVICES=0 python scripts/123_her2st_patient_h_clustering.py \
    --root "$ROOT" \
    --section "$sid" \
    --source measured \
    --top-n 256

  echo "[step] train $sid on 2 GPUs"
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/121_her2st_patient_h_morpho.py \
    --root "$ROOT" \
    --train-slide "$sid" \
    --test-slides "$sid" \
    --mode train \
    --batch-size "$BATCH_SIZE" \
    --predict-batch-size "$PREDICT_BATCH_SIZE" \
    --gene-chunk-size "$GENE_CHUNK_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --epochs "$EPOCHS" \
    --heartbeat-batches "$HEARTBEAT_BATCHES"

  echo "[step] predict $sid"
  CUDA_VISIBLE_DEVICES=0 python scripts/121_her2st_patient_h_morpho.py \
    --root "$ROOT" \
    --train-slide "$sid" \
    --test-slides "$sid" \
    --mode predict \
    --predict-batch-size "$PREDICT_BATCH_SIZE" \
    --gene-chunk-size "$GENE_CHUNK_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --heartbeat-batches "$HEARTBEAT_BATCHES"

  echo "[step] predicted clustering $sid"
  CUDA_VISIBLE_DEVICES=0 python scripts/123_her2st_patient_h_clustering.py \
    --root "$ROOT" \
    --section "$sid" \
    --source pred_bag \
    --train-slide "$sid"

  echo "[step] whole-slide dense $sid"
  CUDA_VISIBLE_DEVICES=0 python scripts/122_her2st_patient_h_wholeslide.py \
    --root "$ROOT" \
    --train-slide "$sid" \
    --section "$sid" \
    --top-n "$WHOLESLIDE_TOP_N" \
    --batch-size "$WHOLESLIDE_BATCH_SIZE"
done

echo "[done] HER2ST patient H pipeline"
