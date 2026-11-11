#!/bin/bash
#SBATCH --job-name=vine_value_bench
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

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
if [ "$ENV_TYPE" = "single_dag" ]; then
    DEFAULT_UPDATES=25
else
    DEFAULT_UPDATES=100
fi
NUM_UPDATES="${2:-${NUM_UPDATES:-$DEFAULT_UPDATES}}"
N_SEEDS="${3:-${N_SEEDS:-5}}"
ALGOS="${4:-${ALGOS:-mc,td_lam0.0,td_lam0.8,vine_k2,vine_k4,fbi_lam0.0,fbi_lam0.8,fbi_staged_lam0.0,fbi_staged_lam0.8}}"
OUT_DIR="${5:-${OUT_DIR:-results}}"
SAVE_DIR="${6:-${SAVE_DIR:-${OUT_DIR}/${ENV_TYPE}_full}}"
MASTER_SEED="${7:-${MASTER_SEED:-42}}"

# Large Reasoning DAG Hyperparameters (leave empty to use smart per-env defaults in run_benchmark.py:
# e.g. single_dag -> M=8, T=32, difficulty=0.5; dag spectrum -> P=10, M=6, T=24)
NUM_PROBLEMS="${NUM_PROBLEMS:-}"
NUM_MILESTONES="${NUM_MILESTONES:-}"
MAX_STEPS="${MAX_STEPS:-}"
NUM_ACTIONS="${NUM_ACTIONS:-8}"
DIFFICULTY="${DIFFICULTY:-0.5}"

# Procedural Tree Hyperparameters
DEPTH="${DEPTH:-7}"
P_CORRECT="${P_CORRECT:-}"
PERCENT_CORRECT="${PERCENT_CORRECT:-}"

# RLVR Optimizer & Schedule Parameters (fixed trunk batch & replay ratio across algorithms)
TRAJECTORIES_PER_UPDATE="${TRAJECTORIES_PER_UPDATE:-64}"
EPOCHS_PER_UPDATE="${EPOCHS_PER_UPDATE:-2}"
GRAD_STEPS_PER_UPDATE="${GRAD_STEPS_PER_UPDATE:-}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"

echo "=================================================================="
echo "Starting Value Learning & Advantage Alignment Benchmark"
echo "  SLURM Job:        ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-${SLURM_JOB_ID:-0}}"
echo "  Host:             $(hostname)"
echo "  Date:             $(date)"
echo "  Python:           $PYTHON"
echo "  Environment:      $ENV_TYPE"
echo "  Updates:          $NUM_UPDATES"
echo "  Seeds:            $N_SEEDS (Master: $MASTER_SEED)"
echo "  Algorithms:       $ALGOS"
echo "  Save Directory:   $SAVE_DIR"
echo "=================================================================="

$PYTHON run_benchmark.py \
    --env_type "$ENV_TYPE" \
    --n_seeds "$N_SEEDS" \
    --master_seed "$MASTER_SEED" \
    --algos "$ALGOS" \
    --out_dir "$OUT_DIR" \
    --save_dir "$SAVE_DIR" \
    ${NUM_PROBLEMS:+--num_problems "$NUM_PROBLEMS"} \
    ${NUM_MILESTONES:+--num_milestones "$NUM_MILESTONES"} \
    ${MAX_STEPS:+--max_steps "$MAX_STEPS"} \
    --num_actions "$NUM_ACTIONS" \
    --difficulty "$DIFFICULTY" \
    --depth "$DEPTH" \
    ${P_CORRECT:+--p_correct "$P_CORRECT"} \
    ${PERCENT_CORRECT:+--percent_correct "$PERCENT_CORRECT"} \
    ${TRAJECTORIES_PER_UPDATE:+--trajectories_per_update "$TRAJECTORIES_PER_UPDATE"} \
    ${EPOCHS_PER_UPDATE:+--epochs_per_update "$EPOCHS_PER_UPDATE"} \
    ${NUM_UPDATES:+--num_updates "$NUM_UPDATES"} \
    ${GRAD_STEPS_PER_UPDATE:+--grad_steps_per_update "$GRAD_STEPS_PER_UPDATE"} \
    ${LR:+--lr "$LR"} \
    ${WEIGHT_DECAY:+--weight_decay "$WEIGHT_DECAY"}

STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "Successfully completed benchmark suite for $ENV_TYPE at $(date)"
else
    echo "Benchmark failed with exit code $STATUS at $(date)"
fi
exit $STATUS
