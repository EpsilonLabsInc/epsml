import argparse
import ast
from io import StringIO
from pprint import pprint

import pandas as pd
from tqdm import tqdm

from epsutils.gcs import gcs_utils


def row_handler(*args):
    row = args[0]
    body_parts_to_include = args[2]
    results = args[3]

    if pd.isna(row["body_part_dicom"]):
        return

    body_part = row["body_part_dicom"].lower()
    body_part_found = False

    for body_part_to_include in body_parts_to_include:
        if body_part_to_include in body_part:
            body_part_found = True
            body_part = body_part_to_include
            break

    if not body_part_found:
        return

    structured_labels = ast.literal_eval(row["structured_labels"])
    labels = [sub_item["label"].lower() for item in structured_labels for sub_item in item["labels"] if item["body_part"] in ["Extremities", "Unknown"]]
    labels = set(labels)

    if body_part not in results:
        results[body_part] = labels
    else:
        results[body_part].update(labels)


def main(args):
    if gcs_utils.is_gcs_uri(args.reports_file):
        print(f"Downloading reports file {args.reports_file}")
        gcs_data = gcs_utils.split_gcs_uri(args.reports_file)
        content = StringIO(gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"]))
    else:
        content = args.reports_file

    print("Loading reports file")
    df = pd.read_csv(content, low_memory=False)

    print("Reading reports file")
    results = {}
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        row_handler(row, index, args.body_parts_to_include, results)

    print("Results:")
    pprint(results, indent=4)


if __name__ == "__main__":
    REPORTS_FILE = r"C:\Users\Andrej\Desktop\gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_all.csv"  # Can be local file or GCS URI.
    BODY_PARTS_TO_INCLUDE = ["leg", "arm", "shoulder", "hand", "wrist", "elbow", "knee", "ankle", "foot", "hip"]

    args = argparse.Namespace(reports_file=REPORTS_FILE,
                              body_parts_to_include=BODY_PARTS_TO_INCLUDE)

    main(args)
