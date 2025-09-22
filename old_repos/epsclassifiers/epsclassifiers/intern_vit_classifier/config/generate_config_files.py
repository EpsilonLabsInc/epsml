import argparse
import copy
import os
import yaml

from epsutils.labels.labels_by_body_part import CONSOLIDATED_LABELS_BY_BODY_PART


def main(args):
    # Check if body part is valid.
    valid_body_parts = CONSOLIDATED_LABELS_BY_BODY_PART.keys()
    if args.body_part not in valid_body_parts:
        response = input(
            f"Body part '{args.body_part}' not valid. Valid body parts are: {', '.join(valid_body_parts)}. "
            "Do you want to continue anyway? [y/n]: ").strip().lower()

        if response != "y":
            print("Exiting without generating config files.")
            exit()

    # Read config template.
    with open(args.config_template, "r") as file:
        content = file.read()
    config = yaml.safe_load(content)

    # Get all labels for the specified body part.
    labels = CONSOLIDATED_LABELS_BY_BODY_PART[args.body_part]

    # Generate config files.
    for label in labels:
        formatted_body_part = args.body_part.lower().replace(' ', '_').replace("/", "_")
        formatted_label = label.lower().replace(' ', '_').replace("/", "_")

        new_config = copy.deepcopy(config)
        new_config["general"]["custom_labels"] = [label]
        new_config["general"]["body_part"] = args.body_part

        output_dir = os.path.join(args.output_dir, formatted_body_part)
        os.makedirs(output_dir, exist_ok=True)

        new_config_file_name = os.path.abspath(
            os.path.join(output_dir, f"{formatted_body_part}_{formatted_label}_config.yaml"))

        print(f"Saving config file {new_config_file_name}")
        with open(new_config_file_name, "w") as f:
            yaml.safe_dump(new_config, f, sort_keys=False)

    print("Generation of config files completed successfully.")


if __name__ == "__main__":
    BODY_PART = "T-spine"
    CONFIG_TEMPLATE = "./v1_3_0/template/v1_3_0_config_template.yaml"
    OUTPUT_DIR = "./v1_3_0"

    args = argparse.Namespace(body_part=BODY_PART,
                              config_template=CONFIG_TEMPLATE,
                              output_dir=OUTPUT_DIR)

    main(args)
