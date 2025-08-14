#!/usr/bin/env bash
set -euo pipefail

# 1) enable nullglob so non-matching globs disappear instead of staying as literal text
shopt -s nullglob

LOG_DIR=./logs
mkdir -p "$LOG_DIR"

# 2) collect all YAMLs under non-chest/<part>/*.yaml
yamls=(config/non-chest/run/*/*.yaml)

# 3) bail if we found nothing
if (( ${#yamls[@]} == 0 )); then
  echo "ERROR: no YAML files found under non-chest/*/*.yaml" >&2
  exit 1
fi

# 4) loop and launch each with a timestamped log
for yaml in "${yamls[@]}"; do
  # skip if somehow it’s not a file
  [[ -f "$yaml" ]] || continue

  part=$(basename "$(dirname "$yaml")")
  fname=$(basename "$yaml" .yaml)
  label=${fname#combined_dataset_${part}_}

  timestamp=$(date +'%Y%m%d_%H%M%S')
  log_file="${LOG_DIR}/${part}_${label}_${timestamp}.log"

  echo "Launching: $yaml → $log_file"
  nohup python run_training_on_combined_dataset.py "$yaml" \
    > "$log_file" 2>&1
done

echo "All jobs launched. Monitor logs in $LOG_DIR/"
