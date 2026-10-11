#!/usr/bin/env bash
set -u
cd /home/huangjinjin/Desktop/Morpho-VC

TRAIN_PID=3487064
TAG=resnet50_pca576_single_INT25_to_INT28
CKPT=checkpoints/st_superres/pathology_encoder_ablation/${TAG}/best_model.pt
LOG=logs/ablation_pathology_encoder_resnet50_single_watcher.log

mkdir -p logs
{
  echo [$(date)] watcher started
  echo [$(date)] train pid: ${TRAIN_PID}
  echo [$(date)] expected checkpoint: ${CKPT}
} >> ${LOG}

while kill -0 ${TRAIN_PID} 2>/dev/null; do
  echo [$(date)] training still running: pid=${TRAIN_PID} >> ${LOG}
  sleep 300
done

sleep 10
if [ ! -s ${CKPT} ]; then
  {
    echo [$(date)] training ended, but best model is missing or empty
    echo [$(date)] skip predict and summary
  } >> ${LOG}
  exit 1
fi

{
  echo [$(date)] training ended and best model exists
  ls -lh ${CKPT}
  echo [$(date)] starting predict
} >> ${LOG}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate morpho-vc
export MORPHO_VC_ROOT=/home/huangjinjin/Desktop/Morpho-VC
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/ablation_pathology_encoder_resnet50_single.py \
  --root /home/huangjinjin/Desktop/Morpho-VC \
  --mode predict \
  --train-slide INT25 \
  --test-slides INT28 \
  --tag ${TAG} \
  --gene-chunk-size 384 \
  --batch-size 4 \
  --predict-batch-size 8 \
  --num-workers 0 \
  --force-predict >> ${LOG} 2>&1
PREDICT_STATUS=$?

if [ ${PREDICT_STATUS} -ne 0 ]; then
  echo [$(date)] predict failed with status ${PREDICT_STATUS}; skip summary >> ${LOG}
  exit ${PREDICT_STATUS}
fi

{
  echo [$(date)] predict finished; starting summary
} >> ${LOG}

python scripts/ablation_pathology_encoder_resnet50_single.py \
  --root /home/huangjinjin/Desktop/Morpho-VC \
  --mode summary \
  --train-slide INT25 \
  --test-slides INT28 \
  --tag ${TAG} >> ${LOG} 2>&1
SUMMARY_STATUS=$?

if [ ${SUMMARY_STATUS} -ne 0 ]; then
  echo [$(date)] summary failed with status ${SUMMARY_STATUS} >> ${LOG}
  exit ${SUMMARY_STATUS}
fi

echo [$(date)] predict and summary completed successfully >> ${LOG}
