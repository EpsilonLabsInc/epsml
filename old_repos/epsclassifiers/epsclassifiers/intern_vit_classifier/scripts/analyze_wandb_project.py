import argparse
import re

import wandb


def main(args):
    api = wandb.Api()
    all_runs = api.runs(f"{args.wandb_entity}/{args.wandb_project}")
    stats = {}

    # Group all runs by body part and label.

    for run in all_runs:
        print(f"Run ID: {run.id}, Name: {run.name}")

        # Parse run name.
        match = re.search(r"combined_([a-zA-Z]+)_([a-zA-Z_]+)-epoch-(\d+)", run.name)
        if not match:
            continue

        # Get body part, label and epoch.
        body_part = match.group(1)
        label = match.group(2)
        epoch = match.group(3)

        if body_part not in stats:
            stats[body_part] = {}

        if label not in stats[body_part]:
            stats[body_part][label] = {"runs": []}

        # Save run.
        stats[body_part][label]["runs"].append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "epoch": epoch,
                "recall": run.summary["Accumulated Recall"],
                "precision": run.summary["Accumulated Precision"]
            }
        )

    # Analyze runs.
    for body_part in stats:
        for label in stats[body_part]:
            runs = stats[body_part][label]["runs"]

            epochs = [int(run["epoch"]) for run in runs]
            recalls = [run["recall"] for run in runs]
            precisions = [run["precision"] for run in runs]

            stats[body_part][label]["num_epochs"] = max(epochs)
            stats[body_part][label]["max_recall"] = max(recalls)
            stats[body_part][label]["max_precision"] = max(precisions)

    # Print stats.
    for body_part in stats:
        print("")
        print("---------------------------------")
        print(f"Body part: {body_part}")
        print("---------------------------------")
        print("")

        for label in stats[body_part]:
            num_epochs = stats[body_part][label]["num_epochs"]
            max_recall = stats[body_part][label]["max_recall"]
            max_precision = stats[body_part][label]["max_precision"]

            print(f"{label}: {max_precision:.2f}, {max_recall:.2f}")


if __name__ == "__main__":
    WANDB_ENTITY = "rustin_r-the-university-of-texas-at-austin"
    WANDB_PROJECT = "intern-vit-classifier-release-version-per-study-non-chest-validation-strategy-max"

    args = argparse.Namespace(wandb_entity=WANDB_ENTITY,
                              wandb_project=WANDB_PROJECT)

    main(args)
