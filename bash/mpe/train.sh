#!/usr/bin/env bash
# OMAR simple_spread / expert: policy stage only.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/settings.sh"

if [[ ! -d "$DATA_PATH" ]]; then
  printf 'Dataset directory not found: %s\nSet OMSD_DATA_PATH to your existing export.\n' "$DATA_PATH" >&2
  exit 2
fi

printf 'Stage outputs: %s/train\n' "$EXPERIMENT_DIR"
exec "$PYTHON_BIN" main.py "${common_args[@]}" --beta "$BETA" \
  --num-steps "$NUM_STEPS" --eval-interval "$EVAL_INTERVAL" \
  --behavior-checkpoint "${BEHAVIOR_CHECKPOINTS[@]}" \
  --critic-checkpoint "${CRITIC_CHECKPOINTS[@]}" \
  --output-dir "$EXPERIMENT_DIR/train" "$@"
