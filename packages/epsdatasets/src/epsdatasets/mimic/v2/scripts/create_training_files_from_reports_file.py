import ast
import json
from io import StringIO

import pandas as pd
from tqdm import tqdm

from epsutils.gcs import gcs_utils

GCS_REPORTS_FILE = "gs://report_csvs/cleaned/CR/MIMIC2_filtered.csv"
TARGET_LABEL = None
OUTPUT_TRAINING_FILE = "mimic-all-labels-training-rui-split.jsonl"
OUTPUT_VALIDATION_FILE = "mimic-all-labels-validation-rui-split.jsonl"


def main():
    # Download reports file.

    print("Downloading reports file")

    gcs_data = gcs_utils.split_gcs_uri(GCS_REPORTS_FILE)
    content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

    # Read reports file.

    print("Reading reports file")

    df = pd.read_csv(StringIO(content), low_memory=False)

    output_training_data = []
    output_validation_data = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        image_paths = ast.literal_eval(row["image_paths"])
        report_text = row["cleaned_report_text"]
        pathologies = ast.literal_eval(row["pathologies"])
        split = row["split"]

        if TARGET_LABEL is not None:
            label = [TARGET_LABEL] if TARGET_LABEL in pathologies else []
        else:
            label = pathologies

        assert split in ("train", "test")

        for image_path in image_paths:
            if split == "train":
                output_training_data.append({"image_path": image_path, "report_text": report_text, "labels": label, "original_labels": pathologies})
            else:
                output_validation_data.append({"image_path": image_path, "report_text": report_text, "labels": label, "original_labels": pathologies})

    # Print statistics.

    if TARGET_LABEL is not None:
        num_positive = 0

        for sample in output_training_data:
            if sample["labels"] == [TARGET_LABEL]:
                num_positive += 1

        print(f"Num {TARGET_LABEL} labels in output training data = {num_positive} / {len(output_training_data)}")

        num_positive = 0

        for sample in output_validation_data:
            if sample["labels"] == [TARGET_LABEL]:
                num_positive += 1

        print(f"Num {TARGET_LABEL} labels in output validation data = {num_positive} / {len(output_validation_data)}")

    # Save output files.

    print("Saving output files")

    with open(OUTPUT_TRAINING_FILE, "w") as f:
        for item in output_training_data:
            f.write(json.dumps(item) + "\n")

    with open(OUTPUT_VALIDATION_FILE, "w") as f:
        for item in output_validation_data:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
