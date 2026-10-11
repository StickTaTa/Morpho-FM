#!/usr/bin/env bash
set -euo pipefail

ROOT="${MORPHO_VC_ROOT:-/share/home/nxy1zll2hjj/hjj/Morpho-VC}"
PYTHON_DEFAULT="/share/home/nxy1zll2hjj/anaconda3/envs/morpho-vc/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_DEFAULT}"
TEST_SLIDE="${1:?Usage: run_prostate_protocol_a_morpho_fold.sh TEST_SLIDE}"
BATCH_SIZE="${PROSTATE_A_BATCH_SIZE:-4}"
GENE_CHUNK_SIZE="${PROSTATE_A_GENE_CHUNK_SIZE:-512}"
PREDICT_BATCH_SIZE="${PROSTATE_A_PREDICT_BATCH_SIZE:-8}"
NPROC="${PROSTATE_A_NPROC:-${SLURM_GPUS_ON_NODE:-1}}"

cd "$ROOT"
export MORPHO_VC_ROOT="$ROOT"

case "$TEST_SLIDE" in
  INT25) VAL_SLIDE="INT26"; TRAIN_DESC="INT27,INT28" ;;
  INT26) VAL_SLIDE="INT27"; TRAIN_DESC="INT25,INT28" ;;
  INT27) VAL_SLIDE="INT28"; TRAIN_DESC="INT25,INT26" ;;
  INT28) VAL_SLIDE="INT25"; TRAIN_DESC="INT26,INT27" ;;
  *) echo "Unknown test slide: $TEST_SLIDE" >&2; exit 2 ;;
esac

TAG="test_${TEST_SLIDE}_val_${VAL_SLIDE}"
LOCK_ROOT="$ROOT/run_locks"
LOCK_DIR="$LOCK_ROOT/prostate_a_${TAG}.lock"
DONE_MARKER="$LOCK_ROOT/prostate_a_${TAG}.done"
mkdir -p "$LOCK_ROOT"

if [[ -f "$DONE_MARKER" ]]; then
  echo "Fold already marked complete, exiting without duplicate compute: $TAG"
  exit 0
fi

set +e
"$PYTHON_BIN" - "$ROOT" "$TAG" "$TEST_SLIDE" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
tag = sys.argv[2]
test_slide = sys.argv[3]
ckpt = root / "checkpoints/st_superres/protocol_a_rotating" / f"cellfm_multi_{tag}" / "best_model.pt"
res = root / "results/prostate_hest/morpho/multi_rotating" / tag / test_slide / "metrics.json"
if ckpt.exists() and res.exists():
    marker = root / "run_locks" / f"prostate_a_{tag}.done"
    marker.write_text("complete\n", encoding="utf-8")
    print(f"Fold already has checkpoint and metrics, marked done: {tag}")
    raise SystemExit(0)
raise SystemExit(1)
PY
status=$?
set -e
if [[ "$status" == "0" ]]; then
  exit 0
fi

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Fold already locked, exiting without duplicate compute: $TAG"
  exit 0
fi
trap 'rm -rf "$LOCK_DIR"' EXIT

PY_CMD=("$PYTHON_BIN" "$ROOT/scripts/prostate_protocol_a_morpho.py")
AGG_CMD=("$PYTHON_BIN" "$ROOT/scripts/aggregate_prostate_protocol_a_morpho.py")

echo "=== Prostate Protocol-A Morpho-FM fold ==="
echo "host=$(hostname)"
echo "root=$ROOT"
echo "test_slide=$TEST_SLIDE"
echo "val_slide=$VAL_SLIDE"
echo "train_slides=$TRAIN_DESC"
echo "tag=$TAG"
echo "batch_size=$BATCH_SIZE"
echo "gene_chunk_size=$GENE_CHUNK_SIZE"
echo "predict_batch_size=$PREDICT_BATCH_SIZE"
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
  "$ROOT/scripts/prostate_protocol_a_morpho.py" \
  --test-slide "$TEST_SLIDE" \
  --val-slide "$VAL_SLIDE" \
  --tag "$TAG" \
  --mode train \
  --batch-size "$BATCH_SIZE" \
  --gene-chunk-size "$GENE_CHUNK_SIZE"

export CUDA_VISIBLE_DEVICES="0"
"${PY_CMD[@]}" \
  --test-slide "$TEST_SLIDE" \
  --val-slide "$VAL_SLIDE" \
  --tag "$TAG" \
  --mode predict \
  --batch-size "$BATCH_SIZE" \
  --predict-batch-size "$PREDICT_BATCH_SIZE" \
  --gene-chunk-size "$GENE_CHUNK_SIZE"

"${PY_CMD[@]}" \
  --test-slide "$TEST_SLIDE" \
  --val-slide "$VAL_SLIDE" \
  --tag "$TAG" \
  --mode aggregate \
  --batch-size "$BATCH_SIZE" \
  --gene-chunk-size "$GENE_CHUNK_SIZE"

"${AGG_CMD[@]}"

touch "$DONE_MARKER"

date
echo "=== Done: $TAG ==="
