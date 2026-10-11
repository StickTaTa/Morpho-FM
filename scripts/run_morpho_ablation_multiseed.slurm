#!/usr/bin/env bash
# Morpho-FM multi-seed decoder-initialization ablation: 20 training runs.
#
# Grid: EXP_MODE {single, multi} x ARM {pretrained, random} x MODEL_SEED {42..46}
#   0- 4  single  pretrained  42..46
#   5- 9  single  random      42..46
#  10-14  multi   pretrained  42..46
#  15-19  multi   random      42..46
#
# SPLIT_SEED is pinned to 42 inside the script, so every run validates on the identical
# blocked spatial partition and ARM is the only variable. All runs share one script
# (notebooks/08_st_superres_train_cellfm_ablation_multiseed.py), fixing the historical
# situation where the two arms were trained by two scripts that had drifted apart.
#
# One run per node on all four A800 GPUs via torchrun, matching how every Morpho-FM run
# has been done. Per the project's remote rules there is no --time and no --mem; the
# partitions are infinite and the node defaults are what we want.
#
#SBATCH -p x86_64_gpu800
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -c 64
#SBATCH --array=0-19
#SBATCH -J morpho_ablation
#SBATCH -o /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/morpho_ablation_%A_%a.log
#SBATCH -e /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/morpho_ablation_%A_%a.log

set -uo pipefail

source /share/home/nxy1zll2hjj/anaconda3/etc/profile.d/conda.sh
conda activate morpho-vc

# 64 cores across 4 ranks. Not the historical AllocCPUS=1, which left the node idle.
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export MORPHO_VC_ROOT=/share/home/nxy1zll2hjj/hjj/Morpho-VC
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

IDX=${SLURM_ARRAY_TASK_ID}

if [ "${IDX}" -lt 10 ]; then EXP_MODE=single; else EXP_MODE=multi; fi
OFF=$(( IDX % 10 ))
if [ "${OFF}" -lt 5 ]; then ARM=pretrained; else ARM=random; fi
MODEL_SEED=$(( 42 + (OFF % 5) ))

export EXP_MODE
export ARM
export MODEL_SEED
export SPLIT_SEED=42

cd "$MORPHO_VC_ROOT"

echo "[task] idx=${IDX} mode=${EXP_MODE} arm=${ARM} seed=${MODEL_SEED} split_seed=${SPLIT_SEED}"
echo "[node] host=$(hostname) gpus=$(nvidia-smi -L | wc -l) cpus=$(nproc) start=$(date -Is)"
nvidia-smi -L

torchrun --standalone --nproc_per_node=4 \
    notebooks/08_st_superres_train_cellfm_ablation_multiseed.py

echo "[done] idx=${IDX} rc=$? $(date -Is)"
