#!/usr/bin/env bash
# Shared settings for MADiff / OG-MARL 2halfcheetah / good (MuJoCo 2.1); source this from the stage scripts.
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "$repo_root"

# Choose the data and training settings once, then use them for every stage.
PYTHON_BIN="${OMSD_PYTHON:-python}"
ALGORITHM="${OMSD_ALGORITHM:-OMSD}"  # OMSD | CTDE | IND
DEVICE="${OMSD_DEVICE:-0}"            # GPU index or cuda:INDEX
SEED="${OMSD_SEED:-42}"               # Model/sampling RNG, not the dataset seed.
DATA_TYPE="${OMSD_DATA_TYPE:-good}"
DATASET_NUM="${OMSD_DATASET_NUM:-0}"   # Recorded dataset label; this export has no seed folders.
DATA_PATH="${OMSD_DATA_PATH:-datasets/ogmarl/2halfcheetah/${DATA_TYPE}}"
# Starting hyperparameters for the OG-MARL good dataset.
TAU="${OMSD_TAU:-.7}"
BETA="${OMSD_BETA:-.01}"
NUM_STEPS="${OMSD_NUM_STEPS:-500000}"
EVAL_INTERVAL="${OMSD_EVAL_INTERVAL:-0}"  # Positive interval enables evaluation.
EXPERIMENT_DIR="${OMSD_OUTPUT_DIR:-outputs/mamujoco210/2halfcheetah/${DATA_TYPE}/${ALGORITHM}/seed${SEED}}"
PRETRAIN_AGENT_ID="${OMSD_PRETRAIN_AGENT_ID:-}"  # Empty trains all learners.

# Final component files are defaults; select intermediate/legacy files here,
# through the OMSD_*_CHECKPOINT overrides, or with train.sh checkpoint arguments.
BEHAVIOR_CHECKPOINTS=("${OMSD_BEHAVIOR_CHECKPOINT:-$EXPERIMENT_DIR/behavior/behavior.pt}")
CRITIC_CHECKPOINTS=("${OMSD_CRITIC_CHECKPOINT:-$EXPERIMENT_DIR/critic/critic.pt}")
common_args=(
  --profile mamujoco210 --env-id 2halfcheetah
  --agent-conf 2x3 --agent-obsk 1
  --algorithm "$ALGORITHM" --device "$DEVICE" --seed "$SEED"
  --data-path "$DATA_PATH" --data-type "$DATA_TYPE" --dataset-num "$DATASET_NUM"
)

# Resolve a command-line agent selector as well as the environment default, so
# independent CLI jobs also get separate output directories.
resolve_pretrain_agent() {
  resolved_agent_id="$PRETRAIN_AGENT_ID"
  while (( $# > 0 )); do
    case "$1" in
      --pretrain-agent-id|--seq-agent-id|--seq_agent_id)
        if (( $# < 2 )); then
          printf 'Missing value for %s\n' "$1" >&2
          return 2
        fi
        resolved_agent_id=$2
        shift 2 ;;
      --pretrain-agent-id=*|--seq-agent-id=*|--seq_agent_id=*)
        resolved_agent_id=${1#*=}
        shift ;;
      *) shift ;;
    esac
  done
}
