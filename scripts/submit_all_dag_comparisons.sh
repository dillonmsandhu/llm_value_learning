#!/usr/bin/env bash
# Submit value-learning and PPO comparisons for all three DAG environments.
set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_BASE=${1:-results/dag_comparisons_v2}
export NUM_PROBLEMS=${NUM_PROBLEMS:-20}
export NUM_MILESTONES=${NUM_MILESTONES:-12}
export MAX_STEPS=${MAX_STEPS:-40}
export N_SEEDS=${N_SEEDS:-5}
export NUM_UPDATES=${NUM_UPDATES:-100}
export MASTER_SEED=${MASTER_SEED:-42}
export NUM_ENVS=${NUM_ENVS:-64}
export BC_UPDATES=${BC_UPDATES:-100}
export EVAL_EPISODES=${EVAL_EPISODES:-1024}
export D_MODEL=${D_MODEL:-64}
export N_LAYERS=${N_LAYERS:-2}

for dag_type in dag dag_quality dag_approach; do
    bash scripts/submit_benchmark.sh "$dag_type" "$NUM_UPDATES" "$N_SEEDS" \
        "$OUTPUT_BASE/value" "mc,td_lam0.8,vine_k2,vine_k4,fbi_lam0.0,fbi_lam0.8" "$MASTER_SEED"
    bash scripts/submit_ppo_comparison.sh "$dag_type" "$OUTPUT_BASE/ppo/$dag_type"
done
