#!/usr/bin/env bash
# Six matched, trunk-only PPO runs; only the critic estimator changes.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: bash scripts/submit_ppo_comparison.sh [DAG_TYPE] [OUTPUT_BASE]

Defaults: dag_approach, results/ppo_comparison/dag_approach.
Submits six GPU jobs: E_min, Vine-target K=2 and K=4, MC, TD(lambda=.8), FBI(lambda=.8,K=2).
E_min uses the bilinear double-sampled energy estimator with one independent
probe per trunk prefix; no critic value clipping is applied to its surrogate.
FBI includes the trunk transition plus one fresh branch. Actor GAE lambda=1
and all other PPO settings are identical across methods.

Base actor configuration from the MC sweep: LR=0.00003, clipping=0.2,
entropy coefficient=0, and 4 actor epochs, shared across every critic method.
Default task/model match that sweep: 20 problems, 12 milestones, 40 steps,
64 rollout environments, 100 PPO updates, 100 BC warmup updates at fixed
BC_LR=0.0003, 64-wide two-layer Transformer, 1024 evaluation episodes.
Runs 5 matched seeds. Critic LR=0.0005 and critic epochs=4 remain fixed.

Shared environment overrides:
  NUM_PROBLEMS, NUM_MILESTONES, MAX_STEPS, NUM_ACTIONS,
  NUM_UPDATES, NUM_ENVS, N_SEEDS, MASTER_SEED, BC_UPDATES, BC_LR, EVAL_EPISODES,
  ACTOR_EPOCHS, CRITIC_EPOCHS, MINIBATCH_SIZE, ACTOR_LR, CRITIC_LR, CLIP_EPS, ENTROPY_COEF,
  D_MODEL, N_HEADS, N_LAYERS, FINAL_HIDDEN_DIM, PYTHON_BIN.

Examples:
  bash scripts/submit_ppo_comparison.sh
  N_SEEDS=1 bash scripts/submit_ppo_comparison.sh dag_approach results/ppo_compare_seed42
  NUM_PROBLEMS=1 NUM_MILESTONES=4 MAX_STEPS=16 BC_UPDATES=20 \
    bash scripts/submit_ppo_comparison.sh dag_quality
  DRY_RUN=1 bash scripts/submit_ppo_comparison.sh

Activate the cluster JAX environment first, or set PYTHON_BIN.
Results go into OUTPUT_BASE/{vine_k2,vine_k4,mc,td_lam08,fbi_lam08_k2,e_min}/.
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
# MC/Vine ignore critic_lambda; MC/TD ignore K. Only Vine K=4 overrides K.
COMMON_ARGS=(
    --num_problems "${NUM_PROBLEMS:-20}"
    --num_milestones "${NUM_MILESTONES:-12}"
    --max_steps "${MAX_STEPS:-40}"
    --num_actions "${NUM_ACTIONS:-8}"
    --num_updates "${NUM_UPDATES:-100}"
    --num_envs "${NUM_ENVS:-64}"
    --n_seeds "${N_SEEDS:-5}"
    --master_seed "${MASTER_SEED:-42}"
    --bc_updates "${BC_UPDATES:-100}"
    --bc_lr "${BC_LR:-0.0003}"
    --eval_episodes "${EVAL_EPISODES:-1024}"
    --actor_epochs "${ACTOR_EPOCHS:-4}"
    --critic_epochs "${CRITIC_EPOCHS:-4}"
    --minibatch_size "${MINIBATCH_SIZE:-64}"
    --actor_lr "${ACTOR_LR:-0.00003}"
    --critic_lr "${CRITIC_LR:-0.0005}"
    --d_model "${D_MODEL:-64}"
    --n_heads "${N_HEADS:-4}"
    --n_layers "${N_LAYERS:-2}"
    --final_hidden_dim "${FINAL_HIDDEN_DIM:-32}"
    --actor_lambda 1.0 --critic_lambda 0.8
    --gamma 1.0 --clip_eps "${CLIP_EPS:-0.2}" --value_clip 0.2
    --entropy_coef "${ENTROPY_COEF:-0}" --max_grad_norm 1.0
)

for variant in vine_k2 vine_k4 mc td fbi e_min; do
    k=2
    method=$variant
    case "$variant" in
        vine_k2) method=vine; label=vine_k2 ;;
        vine_k4) method=vine; label=vine_k4; k=4 ;;
        mc) label=mc ;;
        e_min) label=e_min ;;
        td) label=td_lam08 ;;
        fbi) label=fbi_lam08_k2 ;;
    esac
    command=(bash scripts/submit_dag_ppo.sh "$COMPARE_ENV" "$method"
             "${COMMON_ARGS[@]}" --k "$k" --save_dir "${COMPARE_OUT}/${label}")
    printf '\n%s\n' "PPO comparison: ${label}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}"
    fi
done
