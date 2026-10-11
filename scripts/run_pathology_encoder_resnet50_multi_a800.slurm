#!/bin/bash
#SBATCH -J pathenc_resnet_multi
#SBATCH -p x86_64_gpu800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A800:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH -o logs/pathology_encoder_resnet50_multi_a800_%j.log
#SBATCH -e logs/pathology_encoder_resnet50_multi_a800_%j.err

set -euo pipefail

ROOT="${MORPHO_VC_ROOT:-/share/home/nxy1zll2hjj/hjj/Morpho-VC}"
TAG="${TAG:-resnet50_pca576_multi_INT25_INT26_val_INT27_to_INT28}"
TRAIN_SLIDES="${TRAIN_SLIDES:-INT25,INT26}"
VAL_SLIDES="${VAL_SLIDES:-INT27}"
TEST_SLIDES="${TEST_SLIDES:-INT28}"
GENE_CHUNK_SIZE="${GENE_CHUNK_SIZE:-896}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-100}"
NUM_WORKERS="${NUM_WORKERS:-0}"

export MORPHO_VC_ROOT="$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

source /share/home/nxy1zll2hjj/anaconda3/etc/profile.d/conda.sh
conda activate morpho-vc

cd "$ROOT"
mkdir -p logs

echo "[$(date)] Job start"
echo "Host: $(hostname)"
echo "ROOT=$ROOT"
echo "TAG=$TAG"
echo "TRAIN_SLIDES=$TRAIN_SLIDES VAL_SLIDES=$VAL_SLIDES TEST_SLIDES=$TEST_SLIDES"
echo "GENE_CHUNK_SIZE=$GENE_CHUNK_SIZE BATCH_SIZE=$BATCH_SIZE PREDICT_BATCH_SIZE=$PREDICT_BATCH_SIZE"
nvidia-smi || true

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda devices", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

SCRIPT="scripts/ablation_pathology_encoder_resnet50_multi.py"
COMMON_ARGS=(
  --root "$ROOT"
  --train-slides "$TRAIN_SLIDES"
  --val-slides "$VAL_SLIDES"
  --test-slides "$TEST_SLIDES"
  --tag "$TAG"
  --gene-chunk-size "$GENE_CHUNK_SIZE"
  --batch-size "$BATCH_SIZE"
  --predict-batch-size "$PREDICT_BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --epochs "$EPOCHS"
)

echo "[$(date)] Phase 1/4: build ResNet50 PCA cache"
python "$SCRIPT" --mode cache "${COMMON_ARGS[@]}"

echo "[$(date)] Phase 2/4: train multi-slide ablation with DDP"
torchrun --standalone --nproc_per_node=4 "$SCRIPT" --mode train "${COMMON_ARGS[@]}"

BEST_MODEL="$ROOT/checkpoints/st_superres/pathology_encoder_ablation/$TAG/best_model.pt"
BEST_METRICS="$ROOT/checkpoints/st_superres/pathology_encoder_ablation/$TAG/best_metrics.json"
if [[ ! -s "$BEST_MODEL" ]]; then
  echo "[guard] Training finished but best_model.pt is missing or empty: $BEST_MODEL"
  exit 20
fi
echo "[guard] Found best model: $BEST_MODEL"
if [[ -s "$BEST_METRICS" ]]; then
  echo "[guard] Best metrics:"
  cat "$BEST_METRICS"
fi

echo "[$(date)] Phase 3/4: predict only after best model exists"
python "$SCRIPT" --mode predict "${COMMON_ARGS[@]}"

echo "[$(date)] Phase 4/4: summarize only after prediction"
python "$SCRIPT" --mode summary "${COMMON_ARGS[@]}"

echo "[$(date)] Job complete"
