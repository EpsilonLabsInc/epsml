import ast
import csv
import os

import pandas as pd
from tqdm import tqdm

from epsdatasets.helpers.gradient import gradient_body_parts
from epsdatasets.helpers.gradient.gradient_body_parts import CSV_TO_EPSILON_BODY_PARTS_MAPPING

import config


def main():
    print(f"Reading body parts output file {config.CHECK_BODY_PARTS_OUTPUT_FILE} and generating body parts dictionary")
    body_parts_dict = {}

    with open(config.CHECK_BODY_PARTS_OUTPUT_FILE, "r") as file:
        for line in tqdm(file, desc="Processing"):
            try:
                if line.startswith("ERROR:"):
                    continue

                data = line.split(";")
                assert len(data) == 2

                monai_segmentation_info = ast.literal_eval(data[0].strip())
                file_name = data[1].strip()
                body_parts_dict[file_name] = gradient_body_parts.monai_segmentation_info_to_epsilon_body_parts_distribution(monai_segmentation_info)
            except Exception as e:
                print(f"Error: {str(e)}")

    print(f"Num files found: {len(body_parts_dict)}")

    print("Reading reports file")
    reports = pd.read_csv(config.REPORTS_FILE, sep=",", low_memory=False)

    print("Starting validation")
    validation_data = []
    for index, row in tqdm(reports.iterrows(), total=len(reports), desc="Processing"):
        # Get study data.
        gcs_images_dir = row["GcsImagesDir"]
        primary_volumes = ast.literal_eval(row["PrimaryVolumes"])
        body_parts = ast.literal_eval(row["BodyPartExamined"])

        # Format claimed body parts as a flat list.
        claimed_body_parts = []
        for body_part in body_parts:
            claimed_body_parts.extend([p.strip() for p in body_part.split("/")])
        claimed_body_parts = list(set(claimed_body_parts))
        claimed_body_parts = [CSV_TO_EPSILON_BODY_PARTS_MAPPING[p] for p in claimed_body_parts if p in CSV_TO_EPSILON_BODY_PARTS_MAPPING]

        # Iterate volumes and gather found body parts.
        all_volumes_found = True
        epsilon_body_part_distributions = []
        for primary_volume in primary_volumes:
            full_path = os.path.join(gcs_images_dir, primary_volume)
            if full_path not in body_parts_dict:
                all_volumes_found = False
                break

            epsilon_body_part_distributions.append(body_parts_dict[full_path])

        # Consider only studies with all corresponding volumes found.
        if all_volumes_found and len(primary_volumes) > 0:
            match = gradient_body_parts.match_body_parts(claimed_body_parts, epsilon_body_part_distributions)
            validation_data.append([match, claimed_body_parts, index + 1, primary_volumes])

    print(f"Writing to output file {config.VALIDATION_OUTPUT_FILE}")
    with open(config.VALIDATION_OUTPUT_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Match", "ClaimedBodyParts", "RowNumber", "PrimeryVolumes"])
        writer.writerows(validation_data)


if __name__ == "__main__":
    main()
