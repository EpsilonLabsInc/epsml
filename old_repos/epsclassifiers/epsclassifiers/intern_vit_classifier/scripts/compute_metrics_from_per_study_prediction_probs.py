import argparse
import json
import os
import re
import statistics
from enum import Enum

import torch

from epsutils.gcs import gcs_utils
from epsutils.training.torch_training_helper import TorchTrainingHelper, TrainingParameters, MlopsParameters, MlopsType


class ProbabilitiesReductionStrategy(Enum):
    MAX = 1
    MEAN = 2
    MEDIAN = 3


def probability_reduction(probs, strategy):
    if strategy == ProbabilitiesReductionStrategy.MAX:
        return max(probs)
    elif strategy == ProbabilitiesReductionStrategy.MEAN:
        return statistics.mean(probs)
    elif strategy == ProbabilitiesReductionStrategy.MEDIAN:
        return statistics.median(probs)
    else:
        raise ValueError(f"Unsupported probabilities reduction strategy {strategy}")


def main(args):
    print("Getting a list of per-study prediction probs files")
    assert gcs_utils.is_gcs_uri(args.per_study_prediction_probs_dir)
    gcs_data = gcs_utils.split_gcs_uri(args.per_study_prediction_probs_dir)
    all_files = gcs_utils.list_files(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_dir=gcs_data["gcs_path"], recursive=True)
    pattern = re.compile(r"/per_study_prediction_probs_.*\.jsonl$")
    per_study_prediction_probs_files = [file_name for file_name in all_files if pattern.search(file_name)]

    for index, probs_file in enumerate(per_study_prediction_probs_files):
        print(f"{index + 1}/{len(per_study_prediction_probs_files)} Computing metrics for {probs_file}")

        name = probs_file.split(os.sep)[-3]
        run_name = f"{name}-{str(args.probabilities_reduction_strategy).lower()}"

        mlops_parameters = MlopsParameters(mlops_type=MlopsType.WANDB, experiment_name=args.experiment_name, run_name=run_name,
                                           notes=None, label_names=None, send_notification=False)

        helper = TorchTrainingHelper(model=None, dataset_helper=None, device=None, device_ids=None,
                                     training_parameters=TrainingParameters(), mlops_parameters=mlops_parameters)

        gcs_data = gcs_utils.split_gcs_uri(probs_file)
        content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

        targets = []
        probs = []
        outputs = []

        for line in content.splitlines():
            item = json.loads(line)

            prob = probability_reduction(item["probs"], args.probabilities_reduction_strategy)
            probs.append(prob)

            output = int(prob > 0.5)
            outputs.append(output)

            target = item["target"][0]
            targets.append(target)

        probs = torch.tensor(probs).unsqueeze(1)
        outputs = torch.tensor(outputs).unsqueeze(1)
        targets = torch.tensor(targets).unsqueeze(1)

        helper.compute_metrics(probs=probs, outputs=outputs, targets=targets)


if __name__ == "__main__":
    PER_STUDY_PREDICTION_PROBS_DIR = "gs://epsilonlabs-models/intern-vit-classifier/non-chest"
    EXPERIMENT_NAME = "per-study-non-chest-validation"
    PROBABILITIES_REDUCTION_STRATEGY = ProbabilitiesReductionStrategy.MAX

    args = argparse.Namespace(per_study_prediction_probs_dir=PER_STUDY_PREDICTION_PROBS_DIR,
                              experiment_name=EXPERIMENT_NAME,
                              probabilities_reduction_strategy=PROBABILITIES_REDUCTION_STRATEGY)

    main(args)
