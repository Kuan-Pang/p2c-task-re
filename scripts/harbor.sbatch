#!/usr/bin/env bash
# One Harbor shard on Slurm (PLAN §9). Harbor runs on the node itself: under the `local` profile each
# trial is a host folder (p2c_task_re.harbor_env:LocalEnvironment, no container, no isolation), under
# `singularity` a local container. Tasks shard across Slurm jobs through TASKS, a task-name glob.
#
#   sbatch scripts/harbor.sbatch                                     # oracle over examples/tasks, local
#   DATASET=<dataset> AGENT=p2c_task_re.harbor_agent:SingleShotAgent MODEL=claude-sonnet-5-5 K=20 \
#       TASKS='*-f2-*' sbatch scripts/harbor.sbatch                  # one shard: the F2 tasks
#   PROFILE=singularity sbatch scripts/harbor.sbatch                 # once Apptainer is installed
#   sbatch scripts/harbor.sbatch --ve JUDGE_MODEL=gpt-6.1-sol        # extra arguments go to `harbor run`
#
# Submit from the repo root: relative DATASET and JOBS_DIR resolve against the submit folder, and the Slurm
# log (p2c-harbor-<jobid>.out, gitignored) lands there. The dataset must be packaged for the same profile
# (stages/package.py --environment), because task.toml carries the profile's [environment] table. Regrade a
# finished job with the same environment:
#   uv run harbor job regrade outputs/jobs/<job> -p <dataset> -e p2c_task_re.harbor_env:LocalEnvironment
#
#SBATCH --job-name=p2c-harbor
#SBATCH --partition=il-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

PROFILE="${PROFILE:-local}"        # local | singularity | cloud
DATASET="${DATASET:-examples/tasks}"
AGENT="${AGENT:-oracle}"           # oracle | nop | p2c_task_re.harbor_agent:SingleShotAgent
MODEL="${MODEL:-}"                 # SingleShotAgent only: claude-… or gpt-…
K="${K:-1}"                        # samples per task
N_CONCURRENT="${N_CONCURRENT:-8}"
RETRIES="${RETRIES:-3}"            # only persistent transient errors invalidate a trial
TASKS="${TASKS:-}"                 # task-name glob for this shard; empty = the whole dataset
JOBS_DIR="${JOBS_DIR:-outputs/jobs}"
CLOUD_ENV="${CLOUD_ENV:-modal}"    # the cloud profile's Harbor environment: modal | daytona
API_KEYS="${API_KEYS:-$HOME/.api_keys}"  # exports ANTHROPIC_API_KEY / OPENAI_API_KEY

case "$PROFILE" in
  local) ENVIRONMENT=p2c_task_re.harbor_env:LocalEnvironment ;;
  singularity) ENVIRONMENT=singularity ;;
  cloud) ENVIRONMENT="$CLOUD_ENV" ;;
  *) echo "PROFILE must be local, singularity or cloud, not '$PROFILE'" >&2; exit 1 ;;
esac

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
if [[ -f "$API_KEYS" ]]; then
  source "$API_KEYS"
fi

args=(-p "$DATASET" -e "$ENVIRONMENT" -a "$AGENT" -k "$K" -n "$N_CONCURRENT" -r "$RETRIES" -o "$JOBS_DIR" -y)
if [[ -n "$MODEL" ]]; then args+=(-m "$MODEL"); fi
if [[ -n "$TASKS" ]]; then args+=(-i "$TASKS"); fi

echo "harbor run ${args[*]} $*"
uv run --extra harbor harbor run "${args[@]}" "$@"
