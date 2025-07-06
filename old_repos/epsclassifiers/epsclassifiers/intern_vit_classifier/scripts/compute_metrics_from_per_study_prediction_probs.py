import argparse
import gc
import json
import os
import re

import torch

from epsutils.gcs import gcs_utils
from epsutils.training.torch_training_helper import TorchTrainingHelper, TrainingParameters, MlopsParameters, MlopsType
from epsutils.training.probabilities_reduction import ProbabilitiesReductionStrategy, probabilities_reduction


def main(args):
    print("Getting a list of per-study prediction probs files")
    assert gcs_utils.is_gcs_uri(args.per_study_prediction_probs_dir)
    gcs_data = gcs_utils.split_gcs_uri(args.per_study_prediction_probs_dir)
    all_files = gcs_utils.list_files(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_dir=gcs_data["gcs_path"], recursive=True)
    pattern = re.compile(r"/per_study_prediction_probs_.*\.jsonl$")
    per_study_prediction_probs_files = [file_name for file_name in all_files if pattern.search(file_name)]

    for index, probs_file in enumerate(per_study_prediction_probs_files):
        print(f"{index + 1}/{len(per_study_prediction_probs_files)} Computing metrics for {probs_file}")

        # Get epoch number.
        match = re.search(r"_epoch_(\d+)", probs_file)
        epoch_num = int(match.group(1)) if match else "unknown"

        # Extract the most meaningful part of the file name.
        name = probs_file.split(os.sep)[-3]

        # Generate run name.
        run_name = f"{name}-epoch-{epoch_num}-{str(args.probabilities_reduction_strategy).lower()}"
        print(f"Run name: {run_name}")

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

            prob = probabilities_reduction(item["probs"], args.probabilities_reduction_strategy)
            probs.append(prob)

            output = int(prob > 0.5)
            outputs.append(output)

            target = item["target"][0]
            targets.append(target)

        probs = torch.tensor(probs).unsqueeze(1)
        outputs = torch.tensor(outputs).unsqueeze(1)
        targets = torch.tensor(targets).unsqueeze(1)

        helper.compute_metrics(probs=probs, outputs=outputs, targets=targets)

        del helper
        gc.collect()


if __name__ == "__main__":
    PER_STUDY_PREDICTION_PROBS_DIR = "gs://epsilonlabs-models/intern-vit-classifier/non-chest"
    EXPERIMENT_NAME = "intern-vit-classifier-release-version-per-study-non-chest-validation"
    PROBABILITIES_REDUCTION_STRATEGY = ProbabilitiesReductionStrategy.MAX

    args = argparse.Namespace(per_study_prediction_probs_dir=PER_STUDY_PREDICTION_PROBS_DIR,
                              experiment_name=EXPERIMENT_NAME,
                              probabilities_reduction_strategy=PROBABILITIES_REDUCTION_STRATEGY)

    main(args)
