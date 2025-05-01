import ast
import json
import os
from io import StringIO

import pandas as pd
from tqdm import tqdm

from epsutils.gcs import gcs_utils

GCS_REPORTS_FILE = "gs://report_csvs/cleaned/CR/GRADIENT_CR_ALL_CHEST_BATCHES_cleaned.csv"
ANNOTATION_FILE = "/home/andrej/tmp/pleural_effusion_20250418_tmaung.jsonl"
OUTPUT_ANNOTATION_FILE = "/home/andrej/tmp/pleural_effusion_20250418_tmaung_with_reports.jsonl"


def main():
    print("Downloading reports file")

    # gcs_data = gcs_utils.split_gcs_uri(GCS_REPORTS_FILE)
    # content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

    print("Loading reports file")

    # df = pd.read_csv(StringIO(content), low_memory=False)
    df = pd.read_csv("/home/andrej/tmp/GRADIENT_CR_ALL_CHEST_BATCHES_cleaned.csv", low_memory=False)
    df = df[["report_text", "cleaned_report_text", "image_paths"]]

    print("Creating reports dictionary")

    reports_dict = {}

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        report_text = row["report_text"]
        cleaned_report_text = row["cleaned_report_text"]

        try:
            image_paths = ast.literal_eval(row["image_paths"])
        except Exception as e:
            continue

        if isinstance(image_paths, dict):
            image_paths_dict = image_paths
            image_paths = []
            for value in image_paths_dict.values():
                image_paths.extend(value["paths"])

        image_paths = [os.path.basename(image_path).replace(".dcm", "") for image_path in image_paths]

        for image_path in image_paths:
            reports_dict[image_path] = {"report_text": report_text, "cleaned_report_text": cleaned_report_text}

    print(f"Size of reports dictionary: {len(reports_dict)}")

    print("Loading annotation file")

    annotations = []

    with open(ANNOTATION_FILE, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            annotations.append(data)

    print("Processing annotation file")

    for annotation in annotations:
        file_name = annotation["filename"]
        base_name = file_name.replace(".jpg", "")

        assert base_name in reports_dict

        annotation["report_text"] = reports_dict[base_name]["report_text"]
        annotation["cleaned_report_text"] = reports_dict[base_name]["cleaned_report_text"]

    print("Saving output")

    with open(OUTPUT_ANNOTATION_FILE, "w", encoding="utf-8") as file:
        for annotation in annotations:
            file.write(json.dumps(annotation) + "\n")


if __name__ == "__main__":
    main()
