#!/usr/bin/env bash
# Submit the critic-only K sweep through the existing GPU + plotting jobs.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: bash scripts/submit_k_sweep.sh [DAG_TYPE] [UPDATES] [SEEDS] [LAMBDA] [OUT_DIR] [MASTER_SEED]

Defaults: dag_approach, 100 updates, 8 seeds, lambda=0.8, results/k_sweep_lam0.8, master seed 42.
DAG_TYPE: dag (binary), dag_quality, dag_approach, or program.

Compares TD(lambda) with FBI(lambda), K=1,2,4,8, for both trunk-included
and fresh-branch estimators. Submits one GPU job and its dependent plotting job.
Results: OUT_DIR/DAG_TYPE_full/

Examples:
  bash scripts/submit_k_sweep.sh
  bash scripts/submit_k_sweep.sh dag_quality 100 8 0.8
  bash scripts/submit_k_sweep.sh dag_approach 200 8 1.0 results/k_sweep_lam1

Optional environment overrides: GRAD_STEPS_PER_UPDATE (default 40),
NUM_PROBLEMS (20), NUM_MILESTONES (12), MAX_STEPS (40), plus the settings
already supported by scripts/run_slurm_benchmark.sh.
HELP
    exit 0
fi

cd "$(dirname "$0")/.."
SWEEP_ENV=${1:-dag_approach}
SWEEP_UPDATES=${2:-100}
SWEEP_SEEDS=${3:-8}
SWEEP_LAMBDA=${4:-0.8}
SWEEP_OUT=${5:-results/k_sweep_lam${SWEEP_LAMBDA}}
SWEEP_MASTER_SEED=${6:-42}

case "$SWEEP_ENV" in
    dag|dag_quality|dag_approach|program) ;;
    *) echo "DAG_TYPE must be dag, dag_quality, dag_approach, or program" >&2; exit 2 ;;
esac
if [[ ! "$SWEEP_LAMBDA" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]]; then
    echo "LAMBDA must be between 0 and 1 (for example, 0.8)" >&2
    exit 2
fi
if [[ ! "$SWEEP_UPDATES" =~ ^[1-9][0-9]*$ || ! "$SWEEP_SEEDS" =~ ^[1-9][0-9]*$ || ! "$SWEEP_MASTER_SEED" =~ ^[0-9]+$ ]]; then
    echo "UPDATES and SEEDS must be positive integers; MASTER_SEED must be nonnegative" >&2
    exit 2
fi

SWEEP_ALGOS="td_lam${SWEEP_LAMBDA}"
for k in 1 2 4 8; do
    SWEEP_ALGOS="${SWEEP_ALGOS},fbi_trunk_k${k}_lam${SWEEP_LAMBDA},fbi_k${k}_lam${SWEEP_LAMBDA}"
done

export GRAD_STEPS_PER_UPDATE=${GRAD_STEPS_PER_UPDATE:-40}
export NUM_PROBLEMS=${NUM_PROBLEMS:-20}
export NUM_MILESTONES=${NUM_MILESTONES:-12}
export MAX_STEPS=${MAX_STEPS:-40}

exec bash scripts/submit_benchmark.sh "$SWEEP_ENV" "$SWEEP_UPDATES" \
    "$SWEEP_SEEDS" "$SWEEP_OUT" "$SWEEP_ALGOS" "$SWEEP_MASTER_SEED"
