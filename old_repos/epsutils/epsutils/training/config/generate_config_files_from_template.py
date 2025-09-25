import argparse
import copy
import os
import yaml

from epsutils.labels.labels_by_body_part import CONSOLIDATED_LABELS_BY_BODY_PART


def generate_configs(body_part, config_template, output_dir):
    # Check if body part is valid.
    valid_body_parts = CONSOLIDATED_LABELS_BY_BODY_PART.keys()
    if body_part not in valid_body_parts:
        response = input(
            f"Body part '{body_part}' not valid. Valid body parts are: {', '.join(valid_body_parts)}. "
            "Do you want to continue anyway? [y/n]: ").strip().lower()

        if response != "y":
            print("Exiting without generating config files.")
            exit()

    # Read config template.
    with open(config_template, "r") as file:
        content = file.read()
    config = yaml.safe_load(content)

    # Get all labels for the specified body part.
    labels = CONSOLIDATED_LABELS_BY_BODY_PART[body_part]

    # Generate config files.
    for label in labels:
        formatted_body_part = body_part.lower().replace(' ', '_').replace("/", "_")
        formatted_label = label.lower().replace(' ', '_').replace("/", "_")

        new_config = copy.deepcopy(config)
        new_config["general"]["experiment_name"] += f"___{formatted_body_part}___{formatted_label}"
        new_config["data"]["body_part"] = body_part
        new_config["data"]["custom_labels"] = [label]

        bp_output_dir = os.path.join(output_dir, formatted_body_part)
        os.makedirs(bp_output_dir, exist_ok=True)

        new_config_file_name = os.path.abspath(
            os.path.join(bp_output_dir, f"{formatted_body_part}_{formatted_label}_config.yaml"))

        print(f"Saving config file {new_config_file_name}")
        with open(new_config_file_name, "w") as f:
            yaml.safe_dump(new_config, f, sort_keys=False)


def main(args):
    for index, body_part in enumerate(args.body_parts):
        msg = f"{index + 1}/{len(args.body_parts)} Generating config files for body part {body_part} from template {args.config_template}"
        border = "-" * len(msg)

        print("")
        print(border)
        print(msg)
        print(border)
        print("")

        generate_configs(body_part=body_part, config_template=args.config_template, output_dir=args.output_dir)


if __name__ == "__main__":
    BODY_PARTS = ["Abdomen", "Arm", "C-spine", "Chest", "Foot", "Hand", "Head", "L-spine", "Leg", "Pelvis", "T-spine"]
    CONFIG_TEMPLATE = "./configs/v2_0_0/template/v2_0_0_config_template.yaml"
    OUTPUT_DIR = "./configs/v2_0_0"

    args = argparse.Namespace(body_parts=BODY_PARTS,
                              config_template=CONFIG_TEMPLATE,
                              output_dir=OUTPUT_DIR)

    main(args)
