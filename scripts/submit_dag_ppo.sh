#!/usr/bin/env bash
# Submit PPO on the cluster. Remaining arguments are passed directly to Python.
set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/submit_dag_ppo.sh [dag|dag_quality|dag_approach] [td|mc|fbi|vine] [additional train_dag_ppo.py arguments...]'
    echo 'Example: bash scripts/submit_dag_ppo.sh dag_approach fbi --k 4 --bc_updates 100 --save_dir results/ppo_fbi4'
    exit 0
fi
cd "$(dirname "$0")/.."
PPO_ENV=${1:-dag_approach}
PPO_CRITIC=${2:-td}
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 0 ]]; then shift; fi
case "$PPO_ENV" in dag|dag_quality|dag_approach) ;; *) echo 'Invalid DAG type' >&2; exit 2 ;; esac
case "$PPO_CRITIC" in td|mc|fbi|vine) ;; *) echo 'Invalid critic method' >&2; exit 2 ;; esac
mkdir -p slurm
# %q preserves each argument when SLURM runs the wrapped shell command.
printf -v PPO_COMMAND '%q ' "${PYTHON_BIN:-python3}" -u train_dag_ppo.py \
    --env_type "$PPO_ENV" --critic_method "$PPO_CRITIC" \
    --save_dir "results/ppo_${PPO_ENV}_${PPO_CRITIC}" "$@"
sbatch --job-name="ppo_${PPO_ENV}_${PPO_CRITIC}" \
    --partition=compsci-gpu --gres=gpu:a5000:1 --cpus-per-task=2 \
    --mem=16G --time=02:00:00 --output='slurm/%j_ppo.out' --error='slurm/%j_ppo.err' \
    --export=ALL,XLA_PYTHON_CLIENT_PREALLOCATE=false --wrap="$PPO_COMMAND"
