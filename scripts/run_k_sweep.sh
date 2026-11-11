#!/usr/bin/env bash
# Critic-only comparison on the existing DAG. K counts total backup samples
# for fbi_trunk, and fresh auxiliary samples for fbi_k.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN=${PYTHON_BIN:-python3}
LAMBDA=${LAMBDA:-0.8}
ALGOS="td_lam${LAMBDA}"
for K in 1 2 4 8; do
    ALGOS="${ALGOS},fbi_trunk_k${K}_lam${LAMBDA},fbi_k${K}_lam${LAMBDA}"
done
exec "$PYTHON_BIN" -u run_benchmark.py \
    --env_type "${ENV_TYPE:-dag}" --num_problems 20 --num_milestones 12 --max_steps 40 \
    --num_updates "${NUM_UPDATES:-100}" --grad_steps_per_update 40 \
    --n_seeds "${N_SEEDS:-5}" --algos "$ALGOS" \
    --save_dir "${SAVE_DIR:-results/${ENV_TYPE:-dag}_k_sweep_lam${LAMBDA}}" "$@"
