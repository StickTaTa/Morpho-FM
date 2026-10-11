#!/usr/bin/env bash
set -euo pipefail

ROOT="${MORPHO_VC_ROOT:-/share/home/nxy1zll2hjj/hjj/Morpho-VC}"
PYTHON_DEFAULT="/share/home/nxy1zll2hjj/anaconda3/envs/morpho-vc/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_DEFAULT}"
TRAIN_SLIDE="${1:?Usage: run_prostate_protocol_b_morpho_fold.sh TRAIN_SLIDE}"

cd "$ROOT"
export MORPHO_VC_ROOT="$ROOT"

case "$TRAIN_SLIDE" in
  INT25) TEST_SLIDES="INT26,INT27,INT28" ;;
  INT26) TEST_SLIDES="INT25,INT27,INT28" ;;
  INT27) TEST_SLIDES="INT25,INT26,INT28" ;;
  INT28) TEST_SLIDES="INT25,INT26,INT27" ;;
  *) echo "Unknown train slide: $TRAIN_SLIDE" >&2; exit 2 ;;
esac

TAG="${TRAIN_SLIDE}_to_${TEST_SLIDES//,/_}"
NPROC="${SLURM_GPUS_ON_NODE:-4}"
GENE_CHUNK_SIZE="${GENE_CHUNK_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-8}"

echo "=== Prostate protocol-B Morpho-FM fold ==="
echo "host=$(hostname)"
echo "root=$ROOT"
echo "train_slide=$TRAIN_SLIDE"
echo "test_slides=$TEST_SLIDES"
echo "tag=$TAG"
echo "nproc=$NPROC"
echo "gene_chunk_size=$GENE_CHUNK_SIZE"
echo "batch_size=$BATCH_SIZE"
echo "predict_batch_size=$PREDICT_BATCH_SIZE"
date
nvidia-smi -L

"$PYTHON_BIN" - <<'PY'
import os
import torch
print("python", os.sys.executable)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("root", os.environ.get("MORPHO_VC_ROOT"))
PY

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC" \
  scripts/prostate_protocol_b_morpho.py \
  --train-slide "$TRAIN_SLIDE" \
  --test-slides "$TEST_SLIDES" \
  --tag "$TAG" \
  --batch-size "$BATCH_SIZE" \
  --predict-batch-size "$PREDICT_BATCH_SIZE" \
  --gene-chunk-size "$GENE_CHUNK_SIZE" \
  --mode train

IFS=',' read -r -a TEST_ARRAY <<< "$TEST_SLIDES"
if [ "$NPROC" -lt 1 ]; then
  NPROC=1
fi
gpu=0
active=0
for sid in "${TEST_ARRAY[@]}"; do
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "[predict] slide=$sid gpu=$CUDA_VISIBLE_DEVICES"
    "$PYTHON_BIN" scripts/prostate_protocol_b_morpho.py \
      --train-slide "$TRAIN_SLIDE" \
      --test-slides "$sid" \
      --tag "$TAG" \
      --batch-size "$BATCH_SIZE" \
      --predict-batch-size "$PREDICT_BATCH_SIZE" \
      --gene-chunk-size "$GENE_CHUNK_SIZE" \
      --mode predict
  ) &
  active=$((active + 1))
  gpu=$((gpu + 1))
  if [ "$gpu" -ge "$NPROC" ]; then
    gpu=0
  fi
  if [ "$active" -ge "$NPROC" ]; then
    wait
    active=0
    gpu=0
  fi
done
wait

"$PYTHON_BIN" scripts/prostate_protocol_b_morpho.py \
  --train-slide "$TRAIN_SLIDE" \
  --test-slides "$TEST_SLIDES" \
  --tag "$TAG" \
  --mode aggregate

"$PYTHON_BIN" scripts/aggregate_prostate_protocol_b_morpho.py

date
echo "=== Done: $TAG ==="
