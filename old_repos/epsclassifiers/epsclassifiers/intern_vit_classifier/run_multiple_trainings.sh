#!/bin/bash

# Enable nullglob so non-matching globs disappear instead of staying as literal text.
shopt -s nullglob

# Check if first argument is a directory.
if [ -d "$1" ]; then
  echo "Recursively finding YAML files in: $1"
  mapfile -t config_files < <(find "$1" -type f -name "*.yaml")
  shift
else
  config_files=("$@")
fi

# Check if any config files were found.
if [ "${#config_files[@]}" -eq 0 ]; then
  echo "No config files found."
  exit 1
else
  echo "Found ${#config_files[@]} config file(s):"
  for config_file in "${config_files[@]}"; do
    echo "  - $config_file"
  done
fi

# Loop through the config files and run training.
for config_file in "${config_files[@]}"; do
  python ./run_training_on_combined_dataset.py "$config_file"
done
