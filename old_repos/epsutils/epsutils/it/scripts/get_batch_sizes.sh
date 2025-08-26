#!/bin/bash

BATCH_DIRS=(
  "/mnt/all-data/png/org-size/gradient/09JAN2025"
  "/mnt/all-data/png/org-size/gradient/13JAN2025"
  "/mnt/all-data/png/org-size/gradient/16AUG2024"
  "/mnt/all-data/png/org-size/gradient/20DEC2024"
  "/mnt/all-data/png/org-size/gradient/22JUL2024"
  "/mnt/all-data/png/org-size/segmed/batch1"
  "/mnt/all-data/png/org-size/segmed/batch2"
  "/mnt/all-data/png/org-size/segmed/batch3"
  "/mnt/all-data/png/org-size/segmed/batch4"
  "/mnt/all-data/png/org-size/segmed/batch5"
  "/mnt/all-data/png/org-size/segmed/batch6"
  "/mnt/all-data/png/org-size/segmed/batch7"
  "/mnt/all-data/png/org-size/segmed/batch8"
  "/mnt/all-data/png/org-size/segmed/batch9"
  "/mnt/all-data/png/org-size/segmed/batch10"
  "/mnt/all-data/png/org-size/segmed/batch11"
  "/mnt/all-data/png/org-size/simonmed"
)

echo "Batch disk usage report:"
for dir in "${BATCH_DIRS[@]}"; do
  (
    if [ -d "$dir" ]; then
      usage=$(du -sh "$dir" 2>/dev/null | cut -f1)
      echo "$dir  $usage"
    else
      echo "$dir  Invalid directory"
    fi
  ) &
done

wait
