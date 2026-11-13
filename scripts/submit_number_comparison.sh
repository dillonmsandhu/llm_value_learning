#!/usr/bin/env bash
# Six-method critic comparison on a large search-classified number-pair pool.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: bash scripts/submit_number_comparison.sh [OUTPUT_DIRECTORY]
Default output: results/number_ppo_comparison_<timestamp>
Task: transform a start integer into a target. Default: 37 actions:
{+,-,multiply,exact-divide} x {1,...,9}, plus STOP. STOP submits the current value;
reaching the target alone does not terminate. STOP or the step limit ends the
episode and gives reward 1 iff the current value equals the target, otherwise 0.
Invalid moves consume a step and leave the value unchanged. Default observations
use 54 token IDs, including a distinct STOP token. NUMBER_STOP=0 retains the old
36-action fixed-duration control (no STOP; 53 observation token IDs).

Defaults: domain 0..999, shortest distances 2..5, 8-step budget. Search builds
782,195 training and 195,281 held-out start/target pairs. Reset samples a distance
bucket uniformly, then a pair uniformly. NUM_PROBLEMS does not cap this pool.
NUMBER_SPLIT_MODE=target instead holds entire target numbers out of training.

Submits MC, TD(.8), FBI(.8,K=2), Vine-target K=2/K=4, and E_min.
Shared PPO: actor LR=3e-5, clip=.2, entropy=0, 4 epochs, GAE(1); no KL changes.
Includes held-out pair evaluation and per-distance success/counts.

DRY_RUN=1 prints commands without submission, search, or training.
Overrides: NUMBER_MIN_VALUE, NUMBER_MAX_VALUE, NUMBER_MIN_DISTANCE,
NUMBER_MAX_DISTANCE, NUMBER_TEACHER_PROBABILITY, NUMBER_SPLIT_MODE, NUMBER_STOP;
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
export NUMBER_STOP=${NUMBER_STOP:-1}
case "$NUMBER_STOP" in 0|1) ;; *) echo "NUMBER_STOP must be 0 or 1" >&2; exit 2 ;; esac
export NUM_ACTIONS=$((36+NUMBER_STOP))
exec bash "$(dirname "$0")/submit_ppo_comparison.sh" numbers \
    "${1:-results/number_ppo_comparison_$(date +%Y%m%d_%H%M%S)}"
