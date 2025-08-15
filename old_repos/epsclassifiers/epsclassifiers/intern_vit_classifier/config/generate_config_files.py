import argparse
import copy
import os
import yaml

from epsutils.labels.labels_by_body_part import LABELS_BY_BODY_PART


def main(args):
    # Check if body part is valid.
    valid_body_parts = LABELS_BY_BODY_PART.keys()
    if args.body_part not in valid_body_parts:
        response = input(
            f"Body part '{args.body_part}' not valid. Valid body parts are: {', '.join(valid_body_parts)}. "
            "Do you want to continue anyway? [y/n]: ").strip().lower()

        if response != "y":
            print("Exiting without generating config files.")
            exit()

    # Read config template.
    config_template = args.chest_config_template if args.body_part == "Chest" else args.non_chest_config_template
    with open(config_template, "r") as file:
        content = file.read()
    config = yaml.safe_load(content)

    # Get all labels for the specified body part.
    labels = LABELS_BY_BODY_PART[args.body_part]

    # Generate config files.
    for label in labels:
        formatted_body_part = args.body_part.lower().replace(' ', '_').replace("/", "_")
        formatted_label = label.lower().replace(' ', '_').replace("/", "_")

        new_config = copy.deepcopy(config)
        new_config["general"]["dataset_name"] = f"{formatted_body_part}_{formatted_label}"
        new_config["general"]["run_name"] = args.run_name
        new_config["general"]["notes"] = ""
        new_config["general"]["custom_labels"] = [label]
        new_config["general"]["body_part"] = args.body_part

        if args.num_data_augmentations is not None and label in args.num_data_augmentations:
            new_config["general"]["num_data_augmentations"] = args.num_data_augmentations[label]

        output_dir = os.path.join(args.output_dir, formatted_body_part)
        os.makedirs(output_dir, exist_ok=True)

        new_config_file_name = os.path.abspath(
            os.path.join(output_dir, f"{formatted_body_part}_{formatted_label}_config.yaml"))

        print(f"Saving config file {new_config_file_name}")
        with open(new_config_file_name, "w") as f:
            yaml.safe_dump(new_config, f, sort_keys=False)

    print("Generation of config files completed successfully.")


if __name__ == "__main__":
    BODY_PART = "Chest"
    CHEST_CONFIG_TEMPLATE = "./template/chest_config_template.yaml"
    NON_CHEST_CONFIG_TEMPLATE = "./template/non_chest_config_template.yaml"
    RUN_NAME = "Release models training"
    NUM_DATA_AUGMENTATIONS = None
    # NUM_DATA_AUGMENTATIONS = {
    #     "Adenopathy": 1
    # }
    OUTPUT_DIR = "./generated"

    args = argparse.Namespace(body_part=BODY_PART,
                              chest_config_template=CHEST_CONFIG_TEMPLATE,
                              non_chest_config_template=NON_CHEST_CONFIG_TEMPLATE,
                              run_name=RUN_NAME,
                              num_data_augmentations=NUM_DATA_AUGMENTATIONS,
                              output_dir=OUTPUT_DIR)

    main(args)
