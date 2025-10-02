import argparse
import glob
import json
import os
import re

import torch
from tqdm import tqdm

from epsutils.training.performance_curve_calculator import PerformanceCurveCalculator, PerformanceCurveType


def sort_key(path, delimiter="epoch_"):
    """
    Function assumes path is something like ".../prediction_probs_epoch_1_20250929_162602_utc.jsonl".
    """

    filename = os.path.basename(path)

    if delimiter in filename:
        # Use everything before delimiter as the group.
        group = path.split(delimiter)[0]

        # Try to extract the epoch number.
        match = re.search(rf"{re.escape(delimiter)}(\d+)", filename)

        if match:
            # Numeric sort within the group.
            epoch_num = int(match.group(1))
            return (group, 0, epoch_num)
        else:
            # Fallback to lexicographic sort if epoch number is malformed.
            return (group, 1, filename)
    else:
        # Sort lexicographically if there's no epoch number.
        return (path, 1, filename)


def sort_files(path):
    return sorted(path, key=sort_key)


def main(args):
    # Scan dir for all prediction probs files.
    print(f"Scanning {args.prediction_probs_dir} for all prediction probs files")
    pattern = os.path.join(args.prediction_probs_dir, "**", "prediction_probs_*")
    prediction_probs_files = glob.glob(pattern, recursive=True)

    # Iterate over all files and compute metrics.
    print("Computing metrics")
    performance_curve_calc = PerformanceCurveCalculator()
    metrics = {}
    for prediction_probs_file in tqdm(prediction_probs_files, total=len(prediction_probs_files), desc="Processing"):
        with open(prediction_probs_file, "r", encoding="utf-8") as f:
            content = f.read()

        probs = []
        outputs = []
        targets = []

        for line in content.splitlines():
            item = json.loads(line)
            probs.append(item["prob"])
            outputs.append(item["output"][0])
            targets.append(item["target"][0])

        probs = torch.tensor(probs).unsqueeze(1)
        outputs = torch.tensor(outputs).unsqueeze(1)
        targets = torch.tensor(targets).unsqueeze(1)

        roc = performance_curve_calc.compute_curve(curve_type=PerformanceCurveType.PRC, y_true=targets, y_prob=probs)

        rel_path = os.path.relpath(prediction_probs_file, args.prediction_probs_dir)
        metrics[rel_path] = {"prc_auc": roc["auc"]}

    # Save metrics to a file.
    print(f"Saving metrics to {args.output_file}")
    sorted_keys = sort_files(metrics.keys())
    with open(args.output_file, "w", encoding="utf-8") as f:
        for key in sorted_keys:
            json_line = json.dumps({key: metrics[key]})
            f.write(json_line + "\n")


if __name__ == "__main__":
    PREDICTION_PROBS_DIR = "/mnt/training/v2.0.0/checkpoints"
    OUTPUT_FILE = "metrics.jsonl"

    args = argparse.Namespace(prediction_probs_dir=PREDICTION_PROBS_DIR,
                              output_file=OUTPUT_FILE)

    main(args)
