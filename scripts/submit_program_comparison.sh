#!/usr/bin/env bash
# Restricted program-synthesis PPO comparison, using the shared tuned actor.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: bash scripts/submit_program_comparison.sh [OUTPUT_DIRECTORY]
Default output: results/program_ppo_comparison_<timestamp>
Submits MC, TD(.8), FBI(.8,K=2), Vine-target K=2/K=4, and E_min.
Default task: synthetic integer-polynomial expressions, 2–4 reference operations,
6 input slots, 10 registers (412 actions), degree <=3, coefficient bound 1024,
24-step horizon. Reports training-bank and held-out-problem evaluation separately.
No external data is downloaded. PROGRAM_PROBLEM_FILE selects validated JSONL
made by scripts/prepare_program_tasks.py from a compatible existing Q/A file.
Use the same compilation bounds for import and training.

Shared PPO: actor LR=3e-5, clip=.2, entropy=0, 4 epochs, GAE(1), no KL penalty
or stopping rule; 100 updates, 100 BC warmup updates, 5 seeds. Model: 64-wide,
2 layers. Each setting samples 64 trunk episodes per update.

DRY_RUN=1 prints commands without submitting or training.
Overrides: PROGRAM_MIN_OPS, PROGRAM_MAX_OPS, PROGRAM_DEGREE,
PROGRAM_MAX_INPUTS, PROGRAM_REGISTERS, PROGRAM_COEFFICIENT_LIMIT,
PROGRAM_TEACHER_PROBABILITY, PROGRAM_PROBLEM_FILE; all shared controls from
scripts/submit_ppo_comparison.sh, including N_SEEDS, NUM_UPDATES, MAX_STEPS,
NUM_PROBLEMS, VINE_BATCH_SIZE, ACTOR_LR and PYTHON_BIN.
PROGRAM_REGISTERS must cover PROGRAM_MAX_INPUTS + PROGRAM_MAX_OPS.

Examples:
  DRY_RUN=1 bash scripts/submit_program_comparison.sh results/program_preview
  bash scripts/submit_program_comparison.sh results/program_compare
  PROGRAM_MIN_OPS=4 PROGRAM_MAX_OPS=6 PROGRAM_REGISTERS=12 MAX_STEPS=32 \
    bash scripts/submit_program_comparison.sh results/program_deeper
HELP
    exit 0
fi
if [[ $# -gt 1 ]]; then echo 'Expected at most an output directory; see --help' >&2; exit 2; fi
export MAX_STEPS=${MAX_STEPS:-24}
exec bash "$(dirname "$0")/submit_ppo_comparison.sh" program \
    "${1:-results/program_ppo_comparison_$(date +%Y%m%d_%H%M%S)}"
