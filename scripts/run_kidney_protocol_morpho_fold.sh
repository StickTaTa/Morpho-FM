#!/usr/bin/env bash
set -euo pipefail

ROOT="${MORPHO_VC_ROOT:-/share/home/nxy1zll2hjj/hjj/Morpho-VC}"
PYTHON_DEFAULT="/share/home/nxy1zll2hjj/anaconda3/envs/morpho-vc/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_DEFAULT}"
PROTOCOL="${1:?Usage: run_kidney_protocol_morpho_fold.sh PROTOCOL SLIDE}"
SLIDE="${2:?Usage: run_kidney_protocol_morpho_fold.sh PROTOCOL SLIDE}"
BATCH_SIZE="${KIDNEY_BATCH_SIZE:-4}"
GENE_CHUNK_SIZE="${KIDNEY_GENE_CHUNK_SIZE:-512}"
PREDICT_BATCH_SIZE="${KIDNEY_PREDICT_BATCH_SIZE:-8}"
NPROC="${KIDNEY_NPROC:-${SLURM_GPUS_ON_NODE:-4}}"

cd "$ROOT"
export MORPHO_VC_ROOT="$ROOT"

KIDNEY_SLIDES=(INT13 INT14 INT15 INT17 INT18 INT19 INT21 INT24)

if [[ "$PROTOCOL" == "single_rotating" ]]; then
  TRAIN_SLIDE="$SLIDE"
  TEST_ARRAY=()
  for sid in "${KIDNEY_SLIDES[@]}"; do
    if [[ "$sid" != "$TRAIN_SLIDE" ]]; then
      TEST_ARRAY+=("$sid")
    fi
  done
  TEST_JOINED="$(IFS=_; echo "${TEST_ARRAY[*]}")"
  TAG="train_${TRAIN_SLIDE}_to_${TEST_JOINED}"
  TRAIN_ARGS=(--protocol single_rotating --train-slide "$TRAIN_SLIDE" --tag "$TAG")
else
  echo "Only single_rotating is enabled for this execution phase: $PROTOCOL" >&2
  exit 2
fi

echo "=== Kidney Morpho-FM fold ==="
echo "host=$(hostname)"
echo "root=$ROOT"
echo "protocol=$PROTOCOL"
echo "slide=$SLIDE"
echo "tag=$TAG"
echo "batch_size=$BATCH_SIZE"
echo "gene_chunk_size=$GENE_CHUNK_SIZE"
echo "nproc=$NPROC"
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
  scripts/kidney_protocol_morpho.py \
  "${TRAIN_ARGS[@]}" \
  --mode train \
  --batch-size "$BATCH_SIZE" \
  --gene-chunk-size "$GENE_CHUNK_SIZE"

for ((start=0; start<${#TEST_ARRAY[@]}; start+=NPROC)); do
  for ((j=0; j<NPROC && start+j<${#TEST_ARRAY[@]}; j++)); do
    sid="${TEST_ARRAY[$((start+j))]}"
    (
      export CUDA_VISIBLE_DEVICES="$j"
      "$PYTHON_BIN" scripts/kidney_protocol_morpho.py \
        "${TRAIN_ARGS[@]}" \
        --test-slides "$sid" \
        --mode predict \
        --batch-size "$BATCH_SIZE" \
        --predict-batch-size "$PREDICT_BATCH_SIZE" \
        --gene-chunk-size "$GENE_CHUNK_SIZE"
    ) &
  done
  wait
done

"$PYTHON_BIN" scripts/kidney_protocol_morpho.py \
  "${TRAIN_ARGS[@]}" \
  --mode aggregate \
  --batch-size "$BATCH_SIZE" \
  --gene-chunk-size "$GENE_CHUNK_SIZE"

"$PYTHON_BIN" scripts/aggregate_kidney_protocol_morpho.py

date
echo "=== Done: $PROTOCOL::$TAG ==="
