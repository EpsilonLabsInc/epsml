#!/bin/bash

# Usage:
# ./setup_remote_env.sh ip_list.txt /remote/temp/path username

IP_FILE="$1"
REMOTE_TEMP_PATH="$2"
USERNAME="$3"

# Validate inputs.
if [[ ! -f "$IP_FILE" ]]; then
  echo "IP list file $IP_FILE not found"
  exit 1
fi

if [[ -z "$REMOTE_TEMP_PATH" || -z "$USERNAME" ]]; then
  echo "Missing remote temporary path or username"
  exit 1
fi

# Count total IPs (excluding comments and empty lines)
TOTAL_IPS=$(grep -v '^\s*#' "$IP_FILE" | grep -v '^\s*$' | wc -l)
COUNT=0

# Loop through IPs.
while IFS= read -r line || [[ -n "$line" ]]; do
  # Strip comments and whitespaces.
  IP=$(echo "$line" | sed 's/#.*//' | xargs)

  # Skip empty lines.
  [[ -z "$IP" ]] && continue

  # Print header.
  COUNT=$((COUNT + 1))
  header=">>> REMOTE ENV SETUP FOR $IP [$COUNT/$TOTAL_IPS] <<<"
  width=${#header}
  border=$(printf '=%.0s' $(seq 1 $width))

  echo ""
  echo "$border"
  echo "$header"
  echo "$border"
  echo ""

  # Copy files to remote machine.
  echo "Copying to $IP..."
  rsync -avz --delete . "$USERNAME@$IP:$REMOTE_TEMP_PATH"

  if [[ $? -ne 0 ]]; then
    echo "Copy failed"
    exit 1
  fi

  # Set execute permission for the setup script.
  echo "Setting executable permission for setup_env.sh..."
  ssh "$USERNAME@$IP" "chmod +x '$REMOTE_TEMP_PATH/setup_env.sh'" < /dev/null

  if [[ $? -ne 0 ]]; then
    echo "Failed to set executable permission"
    exit 1
  fi

  # Run setup script.
  echo "Running setup_env.sh..."
  ssh "$USERNAME@$IP" "'$REMOTE_TEMP_PATH/setup_env.sh'" < /dev/null

  if [[ $? -ne 0 ]]; then
    echo "Setup script execution failed"
    exit 1
  fi

  # Add one empty line after each iteration.
  echo

done < "$IP_FILE"
