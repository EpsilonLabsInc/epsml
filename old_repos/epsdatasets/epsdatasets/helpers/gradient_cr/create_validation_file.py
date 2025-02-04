import ast
import json
import os
from io import StringIO

import pandas as pd
from tqdm import tqdm

from epsutils.gcs import gcs_utils

GCS_REPORTS_FILE = "gs://epsilonlabs-filestore/cleaned_CRs/GRADIENT-DATABASE_CR_09JAN2025.csv"
GCS_CHEST_IMAGES_FILE = "gs://gradient-crs/archive/training/chest_files_gradient_all_3_batches.csv"
TARGET_LABEL = "Pneumothorax"
OUTPUT_VALIDATION_FILE = "gradient-crs-09JAN2025-per-study-chest-images-with-pneumothorax-label-validation.jsonl"


def main():
    print(f"Downloading chest images file {GCS_CHEST_IMAGES_FILE}")
    gcs_data = gcs_utils.split_gcs_uri(GCS_CHEST_IMAGES_FILE)
    content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

    print("Generating a list of chest images")
    df = pd.read_csv(StringIO(content), header=None, sep=';')
    chest_images = set(df[0])

    print(f"Downloading GCS reports file {GCS_REPORTS_FILE}")
    gcs_data = gcs_utils.split_gcs_uri(GCS_REPORTS_FILE)
    content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

    print("Reading file")
    df = pd.read_csv(StringIO(content))
    df = df[["labels", "image_paths", "image_base_path"]]
    print(f"{len(df)} total reports in the file")

    print("Generating validation dataset")
    validation_dataset = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        image_base_path = row["image_base_path"]
        image_paths = ast.literal_eval(row["image_paths"])
        image_paths = [os.path.join(image_base_path, image_path) for image_path in image_paths]
        rel_image_paths = [os.path.relpath(image_path, "gs://epsilon-data-us-central1") for image_path in image_paths]

        # At least one study image must be a chest.
        is_chest = any(rel_image_path in chest_images for rel_image_path in rel_image_paths)
        if not is_chest:
            continue

        labels = [label.strip() for label in row["labels"].split(",")]
        validation_dataset.append({"image_path": image_paths, "labels": [TARGET_LABEL] if TARGET_LABEL in labels else []})

    print(f"Validation dataset has {len(validation_dataset)} rows")

    print("Writing validation dataset to file")
    with open(OUTPUT_VALIDATION_FILE, "w") as f:
        for item in validation_dataset:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
