#!/usr/bin/env bash
# Four matched, trunk-only PPO runs; only the critic estimator changes.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: bash scripts/submit_ppo_comparison.sh [DAG_TYPE] [OUTPUT_BASE]

Defaults: dag_approach, results/ppo_comparison/dag_approach.
Submits four GPU jobs: Vine-target K=2, MC, TD(lambda=.8), FBI(lambda=.8,K=2).
FBI includes the trunk transition plus one fresh branch. Actor GAE lambda=1
and all other PPO settings are identical across methods.

Default task/model match the recent single-task MC run: 1 problem, 4 milestones,
16 steps, 32 rollout environments, 100 PPO updates, 20 BC warmup updates,
32-wide one-layer Transformer, 1024 evaluation episodes. Runs 5 matched seeds.

Shared environment overrides:
  NUM_PROBLEMS, NUM_MILESTONES, MAX_STEPS, NUM_ACTIONS,
  NUM_UPDATES, NUM_ENVS, N_SEEDS, MASTER_SEED, BC_UPDATES, EVAL_EPISODES,
  ACTOR_EPOCHS, CRITIC_EPOCHS, MINIBATCH_SIZE, ACTOR_LR, CRITIC_LR,
  D_MODEL, N_HEADS, N_LAYERS, FINAL_HIDDEN_DIM, PYTHON_BIN.

Examples:
  bash scripts/submit_ppo_comparison.sh
  N_SEEDS=1 bash scripts/submit_ppo_comparison.sh dag_approach results/ppo_compare_seed42
  NUM_PROBLEMS=20 NUM_MILESTONES=12 MAX_STEPS=40 BC_UPDATES=100 \
    bash scripts/submit_ppo_comparison.sh dag_quality
  DRY_RUN=1 bash scripts/submit_ppo_comparison.sh

Activate the cluster JAX environment first, or set PYTHON_BIN.
Results go into OUTPUT_BASE/{vine_k2,mc,td_lam08,fbi_lam08_k2}/.
HELP
    exit 0
fi
if [[ $# -gt 2 ]]; then
    echo "Expected at most DAG_TYPE and OUTPUT_BASE; see --help" >&2
    exit 2
fi
cd "$(dirname "$0")/.."
COMPARE_ENV=${1:-dag_approach}
COMPARE_OUT=${2:-results/ppo_comparison/${COMPARE_ENV}}
case "$COMPARE_ENV" in
    dag|dag_quality|dag_approach) ;;
    *) echo "DAG_TYPE must be dag, dag_quality, or dag_approach" >&2; exit 2 ;;
esac

# Keep every setting in one shared array: no per-estimator PPO overrides.
# MC/Vine ignore critic_lambda; MC/TD ignore K. They remain identical in configs.
COMMON_ARGS=(
    --num_problems "${NUM_PROBLEMS:-1}"
    --num_milestones "${NUM_MILESTONES:-4}"
    --max_steps "${MAX_STEPS:-16}"
    --num_actions "${NUM_ACTIONS:-8}"
    --num_updates "${NUM_UPDATES:-100}"
    --num_envs "${NUM_ENVS:-32}"
    --n_seeds "${N_SEEDS:-5}"
    --master_seed "${MASTER_SEED:-42}"
    --bc_updates "${BC_UPDATES:-20}"
    --eval_episodes "${EVAL_EPISODES:-1024}"
    --actor_epochs "${ACTOR_EPOCHS:-4}"
    --critic_epochs "${CRITIC_EPOCHS:-4}"
    --minibatch_size "${MINIBATCH_SIZE:-64}"
    --actor_lr "${ACTOR_LR:-0.0003}"
    --critic_lr "${CRITIC_LR:-0.0005}"
    --d_model "${D_MODEL:-32}"
    --n_heads "${N_HEADS:-4}"
    --n_layers "${N_LAYERS:-1}"
    --final_hidden_dim "${FINAL_HIDDEN_DIM:-32}"
    --actor_lambda 1.0 --critic_lambda 0.8 --k 2
    --gamma 1.0 --clip_eps 0.2 --value_clip 0.2
    --entropy_coef 0.01 --max_grad_norm 1.0
)

for method in vine mc td fbi; do
    case "$method" in
        vine) label=vine_k2 ;;
        mc) label=mc ;;
        td) label=td_lam08 ;;
        fbi) label=fbi_lam08_k2 ;;
    esac
    command=(bash scripts/submit_dag_ppo.sh "$COMPARE_ENV" "$method"
             "${COMMON_ARGS[@]}" --save_dir "${COMPARE_OUT}/${label}")
    printf '\n%s\n' "PPO comparison: ${label}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}"
    fi
done
