import json
import os
import random
from pathlib import Path

import pandas as pd

from epsutils.training.config.config_loader import ConfigLoader


class TrainingJobAnalyzer:
    def __init__(self, ignore_templates=True, low_memory=False, simulation_mode=False, save_intermediate_results_step=None):
        self.__ignore_templates = ignore_templates
        self.__low_memory = low_memory
        self.__simulation_mode = simulation_mode
        self.__save_intermediate_results_step = save_intermediate_results_step

    def find_config_files_and_get_training_jobs(self, root_config_path):
        config_files = list(Path(root_config_path).rglob("*.yaml"))
        config_files = [os.path.abspath(config_file) for config_file in config_files]

        if self.__ignore_templates:
            config_files = [config_file for config_file in config_files if "template" not in config_file.lower()]

        print(f"Found the following {len(config_files)} config files:")
        print(json.dumps(config_files, indent=4))

        return self.get_training_jobs(config_files)

    def get_training_jobs(self, config_files):
        if len(config_files) == 0:
            return []

        # Prefetch the data.
        if not self.__simulation_mode:
            print("Prefetching the data")
            config = ConfigLoader().load_config(config_files[0])
            train_file = config["paths"]["train_file"]
            print(f"Loading {train_file}")
            train_df = pd.read_csv(train_file, low_memory=self.__low_memory)

        training_jobs = []

        for index, config_file in enumerate(config_files):
            # Save intermediate results.
            if self.__save_intermediate_results_step is not None and index % self.__save_intermediate_results_step == 0 and index > 0:
                with open(f"training_jobs_intermediate_{index}.json", "w") as f:
                    json.dump(training_jobs, f, indent=4)

            # Generate status message.
            message = f"{index + 1}/{len(config_files)} Config file: {config_file}"
            border = "=" * len(message)

            # Print status.
            print("")
            print(border)
            print(message)
            print(border)
            print("")

            # Use a random number in simulation mode.
            if self.__simulation_mode:
                training_jobs.append({
                    "job_id": index,
                    "config_file": config_file,
                    "training_dataset_size": None if random.random() < 0.1 else random.randint(10, 200000)
                })
                continue

            # Load config.
            config = ConfigLoader().load_config(config_file)

            # Training files should match across all the configurations.
            if train_file != config["paths"]["train_file"]:
                raise ValueError("Training files should be the same across all the configurations")

            # Create dataset helper.
            try:
                from epsdatasets.helpers.generic.generic_dataset_helper import GenericDatasetHelper
                dataset_helper = GenericDatasetHelper(
                    train_file=config["paths"]["train_file"],
                    validation_file=config["paths"]["validation_file"],
                    test_file=config["paths"]["test_file"],
                    base_path_substitutions=config["paths"]["base_path_substitutions"],
                    body_part=config["data"]["body_part"],
                    sub_body_part=config["data"]["sub_body_part"],
                    merge_val_and_test=True,
                    treat_uncertain_as_positive=config["data"]["treat_uncertain_as_positive"],
                    perform_label_balancing=config["data"]["perform_label_balancing"],
                    max_positive_samples=config["data"]["max_positive_samples"],
                    negative_body_parts_ratio=config["data"]["negative_body_parts_ratio"],
                    num_data_augmentations=config["data"]["num_data_augmentations"],
                    compute_num_data_augmentations=config["data"]["compute_num_data_augmentations"],
                    data_augmentation_target=config["data"]["data_augmentation_target"],
                    data_augmentation_min=config["data"]["data_augmentation_min"],
                    max_study_images=config["data"]["max_study_images"],
                    convert_images_to_rgb=True,
                    replace_dicom_with_png=config["data"]["replace_dicom_with_png"],
                    custom_labels=config["data"]["custom_labels"],
                    train_df=train_df,
                    for_stats_only=True)
            except Exception as e:
                print(e)
                dataset_helper = None

            # Add training job.
            training_jobs.append({
                "job_id": index,
                "config_file": config_file,
                "training_dataset_size": len(dataset_helper.get_torch_train_dataset()) if dataset_helper is not None else None
            })

        return training_jobs


if __name__ == "__main__":
    IGNORE_TEMPLATES = True
    LOW_MEMORY = False
    SIMULATION_MODE = False
    SAVE_INTERMEDIATE_RESULTS_STEP = None
    ROOT_CONFIG_PATH = "../config/configs/v2_0_0"

    training_jobs = TrainingJobAnalyzer(ignore_templates=IGNORE_TEMPLATES,
                                        low_memory=LOW_MEMORY,
                                        simulation_mode=SIMULATION_MODE,
                                        save_intermediate_results_step=SAVE_INTERMEDIATE_RESULTS_STEP).find_config_files_and_get_training_jobs(ROOT_CONFIG_PATH)

    print("Training jobs:")
    print(json.dumps(training_jobs, indent=4))

    print("")
    print("Saving training jobs to file")
    with open("training_jobs.json", "w") as f:
        json.dump(training_jobs, f, indent=4)
