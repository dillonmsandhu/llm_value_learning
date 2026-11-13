#!/usr/bin/env bash
# Plan a pure MC VinePPO sweep, submit the GPU array, then a dependent report job.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: bash scripts/submit_vine_actor_sweep.sh [dag|dag_quality|dag_approach] [OUTPUT]
Default: dag_approach; timestamped results/vine_actor_sweep/... directory.
Pure MC VinePPO K=128, trunk-only PPO; no learned critic.
16 settings: LR={.0001,.0003}, clip={.1,.2}, entropy={0,.001}, epochs={1,4}.
Three matched seeds per setting; 100 updates; 100 BC warmup updates at fixed LR.

Activate your cluster JAX GPU environment first, or set PYTHON_BIN.
DRY_RUN=1 writes the manifest/commands and prints submissions without submitting.
Overrides: ACTOR_LR_GRID, CLIP_EPS_GRID, ENTROPY_COEF_GRID, ACTOR_EPOCHS_GRID
(comma-separated); N_SEEDS, NUM_UPDATES, NUM_ENVS, BC_UPDATES, BC_LR,
NUM_PROBLEMS, NUM_MILESTONES, MAX_STEPS, NUM_ACTIONS, MASTER_SEED,
EVAL_EPISODES, MINIBATCH_SIZE, VINE_BATCH_SIZE, D_MODEL, N_HEADS, N_LAYERS,
FINAL_HIDDEN_DIM; MAX_PARALLEL (4), PARTITION (compsci-gpu), GPU (a5000:1),
MEMORY (24G), CPUS (2), TIME_LIMIT (24:00:00).
The collector runs after all array tasks end, reports missing jobs, and ranks
only complete settings. It inherits the GPU partition and requests one GPU.
HELP
    exit 0
fi
if [[ $# -gt 2 ]]; then echo 'Too many arguments; see --help' >&2; exit 2; fi
cd "$(dirname "$0")/.."
SWEEP_METHOD=${PPO_SWEEP_METHOD:-vine_mc}
case "$SWEEP_METHOD" in
    vine_mc) SWEEP_NAME=vine_actor ;;
    mc) SWEEP_NAME=mc_actor ;;
    *) echo 'PPO_SWEEP_METHOD must be vine_mc or mc' >&2; exit 2 ;;
esac
SWEEP_ENV=${1:-dag_approach}
SWEEP_OUT=${2:-results/${SWEEP_NAME}_sweep/${SWEEP_ENV}_$(date +%Y%m%d_%H%M%S)}
SWEEP_PYTHON=$(command -v "${PYTHON_BIN:-python3}")
export PYTHON_BIN="$SWEEP_PYTHON"
"$SWEEP_PYTHON" vine_actor_sweep.py plan --method "$SWEEP_METHOD" --env "$SWEEP_ENV" --output "$SWEEP_OUT"
SWEEP_OUT=$(cd "$SWEEP_OUT" && pwd)
SWEEP_COUNT=$("$SWEEP_PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["settings"]))' "$SWEEP_OUT/manifest.json")
mkdir -p "$SWEEP_OUT/logs"
printf -v RUN_COMMAND '%q ' "$SWEEP_PYTHON" -u "$PWD/vine_actor_sweep.py" run --output "$SWEEP_OUT"
printf -v COLLECT_COMMAND '%q ' "$SWEEP_PYTHON" -u "$PWD/vine_actor_sweep.py" collect --output "$SWEEP_OUT"
SUBMIT=(sbatch --parsable --chdir="$PWD" --partition="${PARTITION:-compsci-gpu}"
    --gres="gpu:${GPU:-a5000:1}" --cpus-per-task="${CPUS:-2}" --mem="${MEMORY:-24G}"
    --export=ALL,XLA_PYTHON_CLIENT_PREALLOCATE=false,MPLBACKEND=Agg)
ARRAY=("${SUBMIT[@]}" --job-name="${SWEEP_NAME}_sweep" --time="${TIME_LIMIT:-24:00:00}"
    --array="0-$((SWEEP_COUNT-1))%${MAX_PARALLEL:-4}"
    --output="$SWEEP_OUT/logs/%A_%a.out" --error="$SWEEP_OUT/logs/%A_%a.err" --wrap="$RUN_COMMAND")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${ARRAY[@]}"; printf '\n'
    printf '%q ' "${SUBMIT[@]}" '--dependency=afterany:ARRAY_JOB_ID' --job-name="${SWEEP_NAME}_report" \
        --time=00:20:00 --output="$SWEEP_OUT/logs/report_%j.out" --wrap="$COLLECT_COMMAND"; printf '\n'
    exit 0
fi
SWEEP_JOB=$("${ARRAY[@]}")
SWEEP_JOB=${SWEEP_JOB%%;*}
printf '%s\n' "$SWEEP_JOB" > "$SWEEP_OUT/array_job_id.txt"
REPORT_JOB=$("${SUBMIT[@]}" --dependency="afterany:$SWEEP_JOB" --job-name="${SWEEP_NAME}_report" \
    --time=00:20:00 --output="$SWEEP_OUT/logs/report_%j.out" --error="$SWEEP_OUT/logs/report_%j.err" \
    --wrap="$COLLECT_COMMAND")
printf '%s\n' "$REPORT_JOB" > "$SWEEP_OUT/report_job_id.txt"
printf 'Array: %s\nReport job: %s\nResults: %s\n' "$SWEEP_JOB" "$REPORT_JOB" "$SWEEP_OUT"
