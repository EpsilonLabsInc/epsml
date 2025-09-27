#!/bin/bash

# Usage:
# ./run_remote_command.sh ip_list.txt username command

IP_FILE="$1"
USERNAME="$2"
COMMAND="${@:3}"  # Capture full command, including spaces.

# Validate inputs.
if [[ ! -f "$IP_FILE" ]]; then
  echo "IP list file $IP_FILE not found"
  exit 1
fi

if [[ -z "$USERNAME" ]]; then
  echo "Missing username"
  exit 1
fi

if [[ -z "$COMMAND" ]]; then
  echo "Missing command"
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
  header=">>> REMOTE COMMAND EXECUTION FOR $IP [$COUNT/$TOTAL_IPS] <<<"
  width=${#header}
  border=$(printf '=%.0s' $(seq 1 $width))

  echo ""
  echo "$border"
  echo "$header"
  echo "$border"
  echo ""

  # Execute command.
  echo "Executing command on $IP..."
  ssh "$USERNAME@$IP" "$COMMAND" < /dev/null

  if [[ $? -ne 0 ]]; then
    echo "Command execution failed"
    exit 1
  fi

  # Add one empty line after each iteration.
  echo

done < "$IP_FILE"
