import ast
import json
import os
import re
from io import StringIO

import pandas as pd
from tqdm import tqdm

from epsutils.gcs import gcs_utils
from epsutils.labels.cr_chest_labels import CR_CHEST_LABELS

IMAGES_FILE = "gs://gradient-crs/archive/training/self-supervised-training/gradient-crs-all-batches-chest-images-448x448-png.csv"
IMAGES_DIR = "/mnt/efs/all-cxr/gradient-png/"
OUTPUT_IMAGES_FILE = "gradient-crs-all-batches-chest-images-448x448-png-with-labels.json"

"""
List of labels per reports files:

- Alveolar Expanded:
    1 - Airspace Opacity
    2 - Edema
    3 - Consolidation
    4 - Pneumonia
    5 - Lung Lesion
    6 - Atelectasis

- Pleura:
    7 - Pleural Effusion
    8 - Pneumothorax
    9 - Pleural Other

- Cardio:
    10 - Enlarged Cardiomediastinum
    11 - Cardiomegaly

- Support Devices:
    12 - Support Devices

- No Findings:
    13 - No Findings

- Fracture:
    14 - Fracture
"""

REPORTS_DATA = [
    {
        "filename": "gs://report_csvs/cleaned/CR/labels_for_binary_classification/GRADIENT_CR_ALL_CHEST_BATCHES_cleaned_alveolar_expanded_labels.csv",
        "labels_column": "alveolar_expanded_labels"
    },
    {
        "filename": "gs://report_csvs/cleaned/CR/labels_for_binary_classification/GRADIENT_CR_ALL_CHEST_BATCHES_cleaned_pleura_labels.csv",
        "labels_column": "pleura_labels"
    },
    {
        "filename": "gs://report_csvs/cleaned/CR/labels_for_binary_classification/GRADIENT_CR_ALL_CHEST_BATCHES_cleaned_cardio_labels.csv",
        "labels_column": "cardio_labels"
    },
    {
        "filename": "gs://report_csvs/cleaned/CR/labels_for_binary_classification/GRADIENT_CR_ALL_CHEST_BATCHES_cleaned_support_devices_labels.csv",
        "labels_column": "support_devices_labels"
    },
    {
        "filename": "gs://report_csvs/cleaned/CR/labels_for_binary_classification/GRADIENT_CR_ALL_CHEST_BATCHES_cleaned_no_findings_labels.csv",
        "labels_column": "no_findings_labels"
    },
    {
        "filename": "gs://report_csvs/cleaned/CR/labels_for_binary_classification/GRADIENT_CR_ALL_CHEST_BATCHES_cleaned_fracture_labels.csv",
        "labels_column": "fracture_labels"
    }
]


def process_labels(labels):
    labels = set(labels)
    out_labels = set()

    for label in labels:
        label = re.sub(r"\(.*?\)", "", label).strip()

        if label not in CR_CHEST_LABELS:
            continue

        out_labels.add(label)

    return out_labels


def main():
    # Download images file.

    print("Downloading images file")

    gcs_data = gcs_utils.split_gcs_uri(IMAGES_FILE)
    content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

    # Read images file.

    print("Reading images file")

    df = pd.read_csv(StringIO(content), low_memory=False)

    # Create a dict of images.

    df.columns = ["filename"]
    df["filename"] = df["filename"].apply(lambda x: os.path.relpath(x, IMAGES_DIR).replace(".png", ""))
    images = df.set_index("filename").to_dict(orient="index")

    # Print images.

    print(f"Num images: {len(images)}")
    print("Images:")

    for i, (key, value) in enumerate(images.items()):
        if i == 5:
            break

        print(key, ":", value)

    # Download reports files and merge the labels.

    for i, item in enumerate(REPORTS_DATA):
        print("")
        print("--------------------------------")
        print(f"Reports file {i + 1}/{len(REPORTS_DATA)}")
        print("--------------------------------")
        print("")

        reports_file = item["filename"]
        labels_column = item["labels_column"]

        print(f"Downloading reports file {reports_file}")

        gcs_data = gcs_utils.split_gcs_uri(reports_file)
        content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

        print("Reading reports file")

        df = pd.read_csv(StringIO(content), low_memory=False)

        print("Looking for matches")

        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            try:
                batch_id = row["batch_id"]
                if batch_id in ("20DEC2024", "09JAN2025"):
                    batch_id = os.path.join(batch_id, "deid")
                image_paths = ast.literal_eval(row["image_paths"])
                labels = [label.strip() for label in row[labels_column].split(",")]
                labels = process_labels(labels)
            except:
                continue

            if isinstance(image_paths, dict):
                image_paths_dict = image_paths
                image_paths = []
                for value in image_paths_dict.values():
                    image_paths.extend(value["paths"])

            image_paths = [os.path.join(batch_id, image_path).replace(".dcm", "") for image_path in image_paths]

            for image_path in image_paths:
                if not image_path in images:
                    continue

                if "labels" in images[image_path]:
                    images[image_path]["labels"] = images[image_path]["labels"] | labels
                else:
                    images[image_path]["labels"] = set(labels)

    # Print images.

    print("")
    print("Images:")

    for i, (key, value) in enumerate(images.items()):
        if i == 20:
            break

        print(key, ":", value)

    # Save output file.

    print("")
    print("Saving output file")

    with open(OUTPUT_IMAGES_FILE, "w") as f:
        for (key, value) in images.items():
            if not value or "labels" not in value or not value["labels"]:
                continue

            filename = os.path.join(IMAGES_DIR, key) + ".png"
            labels = list(value["labels"])

            if "No Findings" in labels and len(labels) > 1:
                labels.remove("No Findings")

            item = {"filename": filename, "labels": labels}
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
