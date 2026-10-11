#!/usr/bin/env bash
# Audit gate for the STimage row. Runs the vectorization equivalence check at three
# gene counts concurrently, one per GPU, so the A800 node is used rather than held
# for a single sub-minute check.
#
# A mismatch exits non-zero; the row must not be reported if this job fails.
#
# Per the project's remote rules: no --time, no --mem (partitions are infinite and
# node defaults are what we want), and the node is taken exclusively with all GPUs
# and all cores requested and threaded.
#
#SBATCH -p x86_64_gpu800
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -c 64
#SBATCH -J stimage_verify
#SBATCH -o /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/stimage_verify_%j.log
#SBATCH -e /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/stimage_verify_%j.log

set -uo pipefail

source /share/home/nxy1zll2hjj/anaconda3/etc/profile.d/conda.sh
conda activate stimage_bench

export OMP_NUM_THREADS=21
export OPENBLAS_NUM_THREADS=21
export MKL_NUM_THREADS=21
export NUMEXPR_NUM_THREADS=21
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd /share/home/nxy1zll2hjj/hjj/Morpho-VC
mkdir -p logs results/benchmark_audit

echo "[node] $(hostname) gpus=$(nvidia-smi -L | wc -l) cpus=$(nproc) start=$(date -Is)"

GENE_COUNTS=(1 64 1024)
pids=()
for i in "${!GENE_COUNTS[@]}"; do
    g="${GENE_COUNTS[$i]}"
    CUDA_VISIBLE_DEVICES="$i" python -u scripts/verify_stimage_adaptation.py \
        --upstream-root benchmark/STimage \
        --n-genes "$g" \
        --out-json "results/benchmark_audit/stimage_vectorization_check_g${g}.json" \
        > "logs/stimage_verify_g${g}.log" 2>&1 &
    pids+=("$!")
    echo "[launch] n_genes=$g gpu=$i pid=$!"
done

rc=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        rc=1
        echo "[fail] n_genes=${GENE_COUNTS[$i]} exited non-zero"
    fi
done

echo "[node] done=$(date -Is) rc=$rc"
for g in "${GENE_COUNTS[@]}"; do
    echo "=============== n_genes=$g ==============="
    grep -E "^\[exactness\]|^\[timing\]|FAILED|Traceback|AssertionError" "logs/stimage_verify_g${g}.log" || echo "(no summary line)"
done

exit "$rc"
