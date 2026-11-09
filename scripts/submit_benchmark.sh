#!/bin/bash
# ==============================================================================
# scripts/submit_benchmark.sh
#
# Launches the Comprehensive Value Learning Benchmark SLURM Job on the Duke
# CompSci GPU Cluster (compsci-gpu partition, NVIDIA RTX A5000 24GB).
# Automatically chains a dependent CPU post-processing job upon completion.
#
# Usage:
#   bash scripts/submit_benchmark.sh [ENV_TYPE] [TOTAL_TRANSITIONS] [N_SEEDS] [OUT_DIR]
#
# Examples:
#   bash scripts/submit_benchmark.sh dag 250000 5 results
#   bash scripts/submit_benchmark.sh tree 250000 5 results
# ==============================================================================

set -e

[ -f "run_benchmark.py" ] || cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
DATE=$(date +%Y-%m-%d)
SWEEP_NAME="benchmark_value_suite"
LOG_DIR="slurm/${DATE}/${SWEEP_NAME}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# CLI Parameters
ENV_TYPE=${1:-single_dag}
TOTAL_TRANSITIONS=${2:-250000}
N_SEEDS=${3:-5}
OUT_DIR=${4:-"results"}
ALGOS=${5:-"mc,vine_k2,vine_k4,fbi_lam0.0,fbi_staged_lam0.0"}
MASTER_SEED=${6:-42}

# Python interpreter discovery
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/home/users/ds541/.pyenv/shims/python3" ]; then
    PYTHON="/home/users/ds541/.pyenv/shims/python3"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/shims/python3" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/shims/python3"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    PYTHON="python"
fi

SWEEP_ID="bench_${ENV_TYPE}_$(date +%Y%m%d_%H%M%S)"
TARGET_DIR="${OUT_DIR}/benchmark_${ENV_TYPE}_${SWEEP_ID}"

echo "================================================================"
echo "Submitting Comprehensive Value Benchmark to SLURM Cluster"
echo "  Target Env:       $ENV_TYPE"
echo "  Total Budget:     $TOTAL_TRANSITIONS transitions"
echo "  Seeds:            $N_SEEDS (Master: $MASTER_SEED)"
echo "  Algorithms:       $ALGOS"
echo "  Sweep ID:         $SWEEP_ID"
echo "  Expected Output:  $TARGET_DIR"
echo "  Log Directory:    $LOG_DIR"
echo "  Python:           $PYTHON"
echo "  Repo Root:        $REPO_ROOT"
echo "================================================================"

BENCH_SCRIPT="scripts/run_slurm_benchmark.sh"

# 1. Submit GPU Benchmark Job
SBATCH_CMD="sbatch --parsable \
    --job-name=bench_${ENV_TYPE} \
    --output=\"${LOG_DIR}/%j_${ENV_TYPE}.out\" \
    --error=\"${LOG_DIR}/%j_${ENV_TYPE}.err\" \
    --export=ALL,ENV_TYPE=$ENV_TYPE,TOTAL_TRANSITIONS=$TOTAL_TRANSITIONS,N_SEEDS=$N_SEEDS,MASTER_SEED=$MASTER_SEED,ALGOS=\"$ALGOS\",OUT_DIR=\"$OUT_DIR\",SWEEP_ID=\"$SWEEP_ID\",REPO_ROOT=\"$REPO_ROOT\" \
    \"$BENCH_SCRIPT\""

JOB_ID=$(eval "$SBATCH_CMD")
echo "Submitted GPU Benchmark Job ID: $JOB_ID"

# 2. Submit Chained Post-Processing Job (CPU node on partition 'compsci')
POST_SCRIPT="plot_results.py"

SBATCH_POST_CMD="sbatch --parsable \
    --job-name=plot_${ENV_TYPE} \
    --output=\"${LOG_DIR}/%j_plot.out\" \
    --error=\"${LOG_DIR}/%j_plot.err\" \
    --time=00:30:00 \
    --partition=compsci \
    --cpus-per-task=4 \
    --mem=16G \
    --dependency=afterany:$JOB_ID \
    --wrap=\"$PYTHON $POST_SCRIPT --data_path '$TARGET_DIR/benchmark_results.pkl' --save_dir '$TARGET_DIR'\""

POST_JOB_ID=$(eval "$SBATCH_POST_CMD")
echo "Submitted Chained Post-Processing Job ID: $POST_JOB_ID"
echo "  (Will automatically execute after Job $JOB_ID completes)"
echo "Logs written to:    $LOG_DIR/"
echo "Track status with:  squeue -u \$USER"
