#!/usr/bin/env bash
# Feasibility gate for the STimage benchmark row.
#
# Probes upstream `CNN_NB_multiple_genes` at three gene counts so that a failure can
# be attributed to the gene count rather than to the environment:
#     8      -- the regime upstream's own tutorials and HER2ST/OOD scripts actually run
#     1024    -- a typical HVG panel, the scale this repository's other rows use
#     17511   -- this repository's full `data/spatial_data/common_genes.txt` panel
#
# The A800 node is taken exclusively and all four GPUs are requested; the three probes
# run concurrently, one per GPU, so the allocation is actually used rather than held
# while a single tiny probe runs. Per the project's remote rules there is no --time and
# no --mem: the partitions are infinite and the node defaults are what we want.
#
#SBATCH -p x86_64_gpu800
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -c 64
#SBATCH -J stimage_probe
#SBATCH -o /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/stimage_scale_probe_%j.log
#SBATCH -e /share/home/nxy1zll2hjj/hjj/Morpho-VC/logs/stimage_scale_probe_%j.log

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

GENE_COUNTS=(8 1024 17511)
pids=()
for i in "${!GENE_COUNTS[@]}"; do
    n="${GENE_COUNTS[$i]}"
    CUDA_VISIBLE_DEVICES="$i" python -u scripts/probe_stimage_model_scale.py \
        --upstream-root benchmark/STimage \
        --n-genes "$n" \
        --out-json "results/benchmark_audit/stimage_scale_probe_${n}.json" \
        > "logs/stimage_scale_probe_${n}.log" 2>&1 &
    pids+=("$!")
    echo "[launch] n_genes=$n gpu=$i pid=$!"
done

rc=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        rc=1
        echo "[fail] n_genes=${GENE_COUNTS[$i]} exited non-zero"
    fi
done

echo "[node] done=$(date -Is) rc=$rc"
for n in "${GENE_COUNTS[@]}"; do
    echo "=============== n_genes=$n ==============="
    tail -25 "logs/stimage_scale_probe_${n}.log" || true
done

exit "$rc"
