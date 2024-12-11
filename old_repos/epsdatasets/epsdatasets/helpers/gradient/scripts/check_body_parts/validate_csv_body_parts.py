import ast
import csv
import os

import pandas as pd
from tqdm import tqdm

from epsdatasets.helpers.gradient.gradient_body_parts import CSV_TO_EPSILON_BODY_PARTS_MAPPING

import config


def main():
    print("Reading body parts output file")
    data = pd.read_csv(config.CONVERT_TO_EPSILON_BODY_PARTS_OUTPUT_FILE, sep=",", low_memory=False)
    body_parts_dict = data.set_index("Volume")["BodyPart"].to_dict()

    print("Reading reports file")
    reports = pd.read_csv(config.REPORTS_FILE, sep=",", low_memory=False)

    print("Starting validation")
    validation_data = []
    for _, row in tqdm(reports.iterrows(), total=len(reports), desc="Processing"):
        # Get study data.
        row_id = row["row_id"]
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
        found_body_parts = set()
        for primary_volume in primary_volumes:
            full_path = os.path.join(gcs_images_dir, primary_volume)
            if full_path not in body_parts_dict:
                all_volumes_found = False
                break

            found_body_parts.add(body_parts_dict[full_path])

        found_body_parts = list(found_body_parts)

        # Consider only studies with all corresponding volumes found.
        if all_volumes_found and len(primary_volumes) > 0:
            identical = claimed_body_parts == found_body_parts
            validation_data.append([identical, claimed_body_parts, found_body_parts, row_id])

    print(f"Writing to output file {config.VALIDATION_OUTPUT_FILE}")
    with open(config.VALIDATION_OUTPUT_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Identical", "ClaimedBodyParts", "FoundBodyParts", "ReportRowId"])
        writer.writerows(validation_data)


if __name__ == "__main__":
    main()
