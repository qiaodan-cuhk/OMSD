#!/usr/bin/env bash
# OMAR simple_spread / expert: critic stage only.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/settings.sh"

if [[ ! -d "$DATA_PATH" ]]; then
  printf 'Dataset directory not found: %s\nSet OMSD_DATA_PATH to your existing export.\n' "$DATA_PATH" >&2
  exit 2
fi

resolve_pretrain_agent "$@"
output_dir="$EXPERIMENT_DIR/critic"
selection_args=()
if [[ -n "$resolved_agent_id" ]]; then
  output_dir+="/agent$resolved_agent_id"
  selection_args=(--pretrain-agent-id "$resolved_agent_id")
fi
printf 'Stage outputs: %s\n' "$output_dir"
exec "$PYTHON_BIN" pretrain_critic.py "${common_args[@]}" "${selection_args[@]}" --tau "$TAU" \
  --output-dir "$output_dir" "$@"
