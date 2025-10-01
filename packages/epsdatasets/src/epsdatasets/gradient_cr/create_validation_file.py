import ast
import json
import os
from io import StringIO

import pandas as pd
from tqdm import tqdm

from epsutils.gcs import gcs_utils

GCS_REPORTS_FILE = "gs://epsilonlabs-filestore/cleaned_CRs/GRADIENT_CR_batch_1_chest_with_image_paths_with_fracture_labels_structured.csv"
GCS_CHEST_IMAGES_FILE = "gs://gradient-crs/archive/training/chest/chest_files_gradient_all_3_batches.csv"
GCS_IMAGES_DIR = "GRADIENT-DATABASE/CR/22JUL2024"
USE_OLD_REPORT_FORMAT = True
OUTPUT_VALIDATION_FILE = "gradient-crs-22JUL2024-chest-images-with-only-recent-rib-fracture-validation.jsonl"


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
    df = df[["image_paths", "fracture_labels_structured"]]
    print(f"{len(df)} total reports in the file")

    print("Generating validation dataset")
    validation_dataset = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            image_paths = ast.literal_eval(row["image_paths"])
        except Exception as e:
            continue

        if USE_OLD_REPORT_FORMAT:
            image_paths_dict = image_paths
            image_paths = []
            for value in image_paths_dict.values():
                image_paths.extend(value["paths"])

        image_paths = [os.path.join(GCS_IMAGES_DIR, image_path) for image_path in image_paths]

        # At least one study image must be a chest.
        is_chest = all(image_path in chest_images for image_path in image_paths)
        if not is_chest:
            continue

        labels = row["fracture_labels_structured"]
        labels = ast.literal_eval(labels)

        match = False
        for label in labels:
            if label["fracture_type"] == "Recent" and label["body_part"] == "Rib":
                match = True
                break

        if not match:
            continue

        for image_path in image_paths:
            validation_dataset.append({"image_path": image_path, "labels": ["Fracture"]})

    print(f"Validation dataset has {len(validation_dataset)} rows")

    print("Writing validation dataset to file")
    with open(OUTPUT_VALIDATION_FILE, "w") as f:
        for item in validation_dataset:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
