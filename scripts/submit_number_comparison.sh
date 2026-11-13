#!/usr/bin/env bash
# Six-method critic comparison on a large search-classified number-pair pool.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: bash scripts/submit_number_comparison.sh [OUTPUT_DIRECTORY]
Default output: results/number_ppo_comparison_<timestamp>
Task: transform a start integer into a target by the deadline. 36 actions:
{+,-,multiply,exact-divide} x {1,...,9}. No STOP or early termination.
Reward is 1 only if the final value equals the target at the fixed step budget.
The x1 action can preserve an early solution. Invalid moves consume a step and
leave the value unchanged. Decimal digit observations use 53 token IDs.

Defaults: domain 0..999, shortest distances 2..5, 8-step budget. Search builds
782,195 training and 195,281 held-out start/target pairs. Reset samples a distance
bucket uniformly, then a pair uniformly. NUM_PROBLEMS does not cap this pool.
NUMBER_SPLIT_MODE=target instead holds entire target numbers out of training.

Submits MC, TD(.8), FBI(.8,K=2), Vine-target K=2/K=4, and E_min.
Shared PPO: actor LR=3e-5, clip=.2, entropy=0, 4 epochs, GAE(1); no KL changes.
Includes held-out pair evaluation and per-distance success/counts.

DRY_RUN=1 prints commands without submission, search, or training.
Overrides: NUMBER_MIN_VALUE, NUMBER_MAX_VALUE, NUMBER_MIN_DISTANCE,
NUMBER_MAX_DISTANCE, NUMBER_TEACHER_PROBABILITY, NUMBER_SPLIT_MODE;
MAX_STEPS, N_SEEDS, NUM_UPDATES, BC_UPDATES, NUM_ENVS, PYTHON_BIN,
and the other controls of scripts/submit_ppo_comparison.sh.

Examples:
  DRY_RUN=1 bash scripts/submit_number_comparison.sh results/numbers_preview
  bash scripts/submit_number_comparison.sh results/numbers_compare
  NUMBER_SPLIT_MODE=target bash scripts/submit_number_comparison.sh results/numbers_new_targets
HELP
    exit 0
fi
if [[ $# -gt 1 ]]; then echo 'Expected at most an output directory; see --help' >&2; exit 2; fi
export MAX_STEPS=${MAX_STEPS:-8}
export NUM_ACTIONS=36
exec bash "$(dirname "$0")/submit_ppo_comparison.sh" numbers \
    "${1:-results/number_ppo_comparison_$(date +%Y%m%d_%H%M%S)}"
