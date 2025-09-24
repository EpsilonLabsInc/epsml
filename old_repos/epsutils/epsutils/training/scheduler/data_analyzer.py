import json
import os
from pathlib import Path

from epsdatasets.helpers.generic.generic_dataset_helper import GenericDatasetHelper
from epsutils.training.config.config_loader import ConfigLoader


class DataAnalyzer:
    def __init__(self, ignore_templates=True):
        self.__ignore_templates = ignore_templates

    def find_config_files_and_get_num_training_samples(self, root_config_path):
        config_files = list(Path(root_config_path).rglob("*.yaml"))
        config_files = [os.path.abspath(config_file) for config_file in config_files]

        if self.__ignore_templates:
            config_files = [config_file for config_file in config_files if "template" not in config_file.lower()]

        print("Found the following config files:")
        print(json.dumps(config_files, indent=4))

        return self.get_num_training_samples(config_files)

    def get_num_training_samples(self, config_files):
        num_samples_dict = {}

        for index, config_file in enumerate(config_files):
            # Generate status message.
            message = f"{index + 1}/{len(config_files)} Config file: {config_file}"
            border = "=" * len(message)

            # Print status.
            print("")
            print(border)
            print(message)
            print(border)
            print("")

            # Load config.
            config = ConfigLoader().load_config(config_file)

            # Create dataset helper.
            try:
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
                    negative_body_parts_ratio=config["data"]["negative_body_parts_ratio"],
                    num_data_augmentations=config["data"]["num_data_augmentations"],
                    compute_num_data_augmentations=config["data"]["compute_num_data_augmentations"],
                    data_augmentation_target=config["data"]["data_augmentation_target"],
                    data_augmentation_min=config["data"]["data_augmentation_min"],
                    max_study_images=config["data"]["max_study_images"],
                    convert_images_to_rgb=True,
                    replace_dicom_with_png=config["data"]["replace_dicom_with_png"],
                    custom_labels=config["data"]["custom_labels"])
            except Exception as e:
                print(e)
                dataset_helper = None

            # Get num training samples.
            num_samples_dict[config_file] = len(dataset_helper.get_torch_train_dataset()) if dataset_helper is not None else None

        return num_samples_dict


if __name__ == "__main__":
    IGNORE_TEMPLATES = True
    ROOT_CONFIG_PATH = "../config/configs/v1_3_0"

    num_training_samples = DataAnalyzer(ignore_templates=IGNORE_TEMPLATES).find_config_files_and_get_num_training_samples(ROOT_CONFIG_PATH)

    print("Num training samples:")
    print(json.dumps(num_training_samples, indent=4))
