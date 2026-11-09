#!/bin/bash
#SBATCH --job-name=vine_value_bench
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# ==============================================================================
# Comprehensive Value Learning Benchmark: SLURM Task
# Runs all value learning algorithms on the designated environment:
#   1. Standard Monte Carlo Critic (K=1)
#   2. Vine-Target Critic (K=2)
#   3. Vine-Target Critic (K=4)
#   4. Batched Fitted Backward Induction FBI (lambda=0.0)
#   5. Staged Fitted Backward Induction FBI (lambda=0.0)
#   6. Batched Fitted Backward Induction FBI (lambda=0.5)
#
# Metrics Evaluated:
#   - MSVE vs Gradient Updates, Sample Budget, Wall-Clock Compute
#   - Advantage Cosine Similarity vs Updates, Samples, Wall-Clock Compute
#   - True vs Estimated Value Scatter Calibration (at 1/5 and 4/5 of learning)
#   - Cross-Algorithm Aggregate AUC Summary Table
# ==============================================================================

set -e
mkdir -p slurm

# Working directory resolution (handles local shell, OOD interactive terminals, and batch jobs)
if [ -n "$SLURM_SUBMIT_DIR" ] && [ -d "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
[ -f "run_benchmark.py" ] || cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Python interpreter discovery (Duke CompSci GPU cluster & local fallbacks)
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

# Environment Configuration (Accept positional CLI arguments or environment variables)
ENV_TYPE="${1:-${ENV_TYPE:-single_dag}}"
TOTAL_TRANSITIONS="${2:-${TOTAL_TRANSITIONS:-250000}}"
N_SEEDS="${3:-${N_SEEDS:-5}}"
ALGOS="${4:-${ALGOS:-mc,vine_k2,vine_k4,fbi_lam0.0,fbi_staged_lam0.0}}"
OUT_DIR="${5:-${OUT_DIR:-results}}"
SWEEP_ID="${6:-${SWEEP_ID:-}}"
MASTER_SEED="${7:-${MASTER_SEED:-42}}"

# Large Reasoning DAG Hyperparameters
NUM_PROBLEMS="${NUM_PROBLEMS:-10}"
NUM_MILESTONES="${NUM_MILESTONES:-6}"
MAX_STEPS="${MAX_STEPS:-24}"
NUM_ACTIONS="${NUM_ACTIONS:-8}"
DIFFICULTY="${DIFFICULTY:-0.5}"

# Large Tree MRP Hyperparameters
DEPTH="${DEPTH:-7}"
P_SUCCESS="${P_SUCCESS:-0.8}"

# Sweep / Run ID
SWEEP_ID="${SWEEP_ID:-${SLURM_ARRAY_JOB_ID:+bench_${SLURM_ARRAY_JOB_ID}}}"
SWEEP_ID="${SWEEP_ID:-bench_${SLURM_JOB_ID:-$(date +"%Y%m%d_%H%M%S")}}"

echo "=================================================================="
echo "Starting Value Learning & Advantage Alignment Benchmark"
echo "  SLURM Job:        ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-${SLURM_JOB_ID:-0}}"
echo "  Host:             $(hostname)"
echo "  Date:             $(date)"
echo "  Python:           $PYTHON"
echo "  Environment:      $ENV_TYPE"
echo "  Total Budget:     $TOTAL_TRANSITIONS transitions"
echo "  Seeds:            $N_SEEDS (Master: $MASTER_SEED)"
echo "  Algorithms:       $ALGOS"
echo "  Sweep / Run ID:   $SWEEP_ID"
echo "  Output Directory: $OUT_DIR"
echo "=================================================================="

$PYTHON run_benchmark.py \
    --env_type "$ENV_TYPE" \
    --total_transitions "$TOTAL_TRANSITIONS" \
    --n_seeds "$N_SEEDS" \
    --master_seed "$MASTER_SEED" \
    --algos "$ALGOS" \
    --out_dir "$OUT_DIR" \
    --sweep_id "$SWEEP_ID" \
    --num_problems "$NUM_PROBLEMS" \
    --num_milestones "$NUM_MILESTONES" \
    --max_steps "$MAX_STEPS" \
    --num_actions "$NUM_ACTIONS" \
    --difficulty "$DIFFICULTY" \
    --depth "$DEPTH" \
    --p "$P_SUCCESS"

STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "Successfully completed benchmark suite for $ENV_TYPE at $(date)"
else
    echo "Benchmark failed with exit code $STATUS at $(date)"
fi
exit $STATUS
