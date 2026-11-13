#!/usr/bin/env bash
# Ordinary MC PPO actor tuning: full trunk returns, low actor learning rates.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: bash scripts/submit_mc_actor_sweep.sh [dag|dag_quality|dag_approach|program|numbers] [OUTPUT]
Default: dag_approach; timestamped results/mc_actor_sweep/... directory.

Runs the existing MC PPO method: critic targets are full trunk Monte Carlo
returns, actor GAE(lambda=1) is the full return minus the frozen critic baseline.
No vine rollouts; no bootstrapped MC targets. A learned critic provides the
baseline and is fitted to MC returns, with fixed settings across the sweep.

Low-LR grid (32 settings × 3 matched seeds):
  Actor LR:       0.000001, 0.000003, 0.00001, 0.00003
  PPO clipping:   0.1, 0.2
  Entropy coef:   0, 0.001
  Actor epochs:   1, 4

Shared defaults match the Vine sweep: 20 problems, 12 milestones, 40 steps,
64 rollout episodes/update, 100 PPO updates, 100 BC warmup updates at fixed
BC_LR=0.0003, 1024 evaluation episodes, 64-wide two-layer Transformer.
Critic LR=0.0005 and critic epochs=4 are held fixed (CRITIC_LR/CRITIC_EPOCHS).
Four array jobs run concurrently. Defaults request A5000, 24G, 24 hours/job.

Activate the cluster JAX GPU environment first, or set PYTHON_BIN.
Example:
  bash scripts/submit_mc_actor_sweep.sh dag_approach results/mc_low_lr
Preview without submitting or training:
  DRY_RUN=1 bash scripts/submit_mc_actor_sweep.sh dag_approach results/mc_preview
Override the learning-rate grid:
  ACTOR_LR_GRID=0.000003,0.00001,0.00003,0.0001 bash scripts/submit_mc_actor_sweep.sh

Also accepts CLIP_EPS_GRID, ENTROPY_COEF_GRID, ACTOR_EPOCHS_GRID, N_SEEDS,
NUM_UPDATES, BC_UPDATES, BC_LR, NUM_ENVS, MASTER_SEED, EVAL_EPISODES,
NUM_PROBLEMS, NUM_MILESTONES, MAX_STEPS, NUM_ACTIONS, MINIBATCH_SIZE,
D_MODEL, N_HEADS, N_LAYERS, FINAL_HIDDEN_DIM, MAX_PARALLEL, PARTITION,
GPU, MEMORY, CPUS and TIME_LIMIT. Use comma-separated values for grid overrides.

A dependent report job plots all complete learning curves together and writes
ranking_auc.csv, ranking_final_return.csv, per_seed.csv and report.md.
Final return uses fresh final-policy evaluation; AUC uses recorded training
returns. Incomplete settings are listed in collection_status.json and excluded.
Each sweep needs a new output directory. No jobs are submitted by --help.
HELP
    exit 0
fi
export PPO_SWEEP_METHOD=mc
export ACTOR_LR_GRID=${ACTOR_LR_GRID:-0.000001,0.000003,0.00001,0.00003}
exec bash "$(dirname "$0")/submit_vine_actor_sweep.sh" "$@"
