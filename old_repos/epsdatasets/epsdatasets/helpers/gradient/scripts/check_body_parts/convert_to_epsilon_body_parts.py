import ast
import csv

from tqdm import tqdm

from epsdatasets.helpers.gradient.gradient_body_parts import MONAI_TO_EPSILON_BODY_PARTS_MAPPING

import config


def main():
    print(f"Reading input file {config.CHECK_BODY_PARTS_OUTPUT_FILE}")

    out_data = []
    with open(config.CHECK_BODY_PARTS_OUTPUT_FILE, "r") as file:
        for line in tqdm(file, desc="Processing"):
            if line.startswith("ERROR:"):
                continue

            labels, file_name = line.strip().rsplit("},", 1)
            labels += "}"
            labels = ast.literal_eval(labels)

            if labels["top_label"] is None:
                continue

            top_label = MONAI_TO_EPSILON_BODY_PARTS_MAPPING[labels["top_label"]]
            out_data.append([file_name, top_label])

    print(f"Writing to output file {config.CONVERT_TO_EPSILON_BODY_PARTS_OUTPUT_FILE}")

    with open(config.CONVERT_TO_EPSILON_BODY_PARTS_OUTPUT_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Volume", "BodyPart"])
        writer.writerows(out_data)


if __name__ == "__main__":
    main()
