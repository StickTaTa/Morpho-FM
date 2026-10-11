#!/usr/bin/env bash
#SBATCH -p x86_64_gpu800
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -J kidney_pred_only
#SBATCH -o /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/kidney_predict_only_%A.log
#SBATCH -e /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/kidney_predict_only_%A.log

set -euo pipefail
source /share/home/nxy1zll2hjj/anaconda3/etc/profile.d/conda.sh
conda activate morpho-vc

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export MORPHO_VC_ROOT=/share/home/nxy1zll2hjj/hjj/Morpho-VC
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/share/home/nxy1zll2hjj/anaconda3/envs/morpho-vc/bin/python}"
TRAIN_SLIDE="${KIDNEY_TRAIN_SLIDE:?Set KIDNEY_TRAIN_SLIDE}"
TAG="${KIDNEY_TAG:?Set KIDNEY_TAG}"
TEST_SLIDES="${KIDNEY_TEST_SLIDES:?Set KIDNEY_TEST_SLIDES as a space-separated list}"
BATCH_SIZE="${KIDNEY_BATCH_SIZE:-4}"
GENE_CHUNK_SIZE="${KIDNEY_GENE_CHUNK_SIZE:-512}"
PREDICT_BATCH_SIZE="${KIDNEY_PREDICT_BATCH_SIZE:-8}"
NPROC="${KIDNEY_NPROC:-${SLURM_GPUS_ON_NODE:-1}}"

cd /share/home/nxy1zll2hjj/hjj/Morpho-VC
mkdir -p logs

CHECKPOINT="checkpoints/kidney_hest/st_superres/protocol_rotating/single_rotating/${TAG}/best_model.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 3
fi

read -r -a TEST_ARRAY <<< "$TEST_SLIDES"

echo "=== Kidney Morpho-FM predict-only ==="
echo "host=$(hostname)"
echo "train_slide=$TRAIN_SLIDE"
echo "tag=$TAG"
echo "test_slides=$TEST_SLIDES"
echo "batch_size=$BATCH_SIZE"
echo "gene_chunk_size=$GENE_CHUNK_SIZE"
echo "predict_batch_size=$PREDICT_BATCH_SIZE"
echo "nproc=$NPROC"
date
nvidia-smi -L

for ((start=0; start<${#TEST_ARRAY[@]}; start+=NPROC)); do
  for ((j=0; j<NPROC && start+j<${#TEST_ARRAY[@]}; j++)); do
    sid="${TEST_ARRAY[$((start+j))]}"
    (
      export CUDA_VISIBLE_DEVICES="$j"
      "$PYTHON_BIN" scripts/kidney_protocol_morpho.py \
        --protocol single_rotating \
        --train-slide "$TRAIN_SLIDE" \
        --tag "$TAG" \
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
  --protocol single_rotating \
  --train-slide "$TRAIN_SLIDE" \
  --tag "$TAG" \
  --mode aggregate \
  --batch-size "$BATCH_SIZE" \
  --gene-chunk-size "$GENE_CHUNK_SIZE"

"$PYTHON_BIN" scripts/aggregate_kidney_protocol_morpho.py

date
echo "=== Done: kidney predict-only ${TAG} ==="
