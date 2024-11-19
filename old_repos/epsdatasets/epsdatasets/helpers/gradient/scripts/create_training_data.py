import ast
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from google.cloud import bigquery, storage
from tqdm import tqdm

from epsdatasets.helpers.gradient.gradient_labels import GROUPED_GRADIENT_LABELS
from epsutils.gcs import gcs_utils
from epsutils.labels.grouped_labels_manager import GroupedLabelsManager

DATALAKE_QUERY = """
SELECT primary_volumes, structured_labels, gcs_bucket_name, gcs_images_dir
FROM `datalake.CT-2024-11-18`
WHERE body_part = "['Chest']" AND study_accepted = true
"""

CONTENT_TO_SEARCH = "(0018,0015) Body Part Examined: Chest\n"
OUTPUT_FILE = "ct_chest_only_training_data.csv"


def process_row(row):
    nifti_files = row["nifti_files"]
    labels = row["labels"]
    gcs_bucket_name = row["gcs_bucket_name"]
    gcs_images_dir = row["gcs_images_dir"]

    chest_nifti_files = []

    client = storage.Client()
    bucket = client.get_bucket(gcs_bucket_name)

    for nifti_file in nifti_files:
        txt_file = os.path.join(gcs_images_dir, nifti_file).replace(".nii.gz", ".txt")

        try:
            dicom_content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_bucket_name, gcs_file_name=txt_file)
        except:
            continue

        if CONTENT_TO_SEARCH in dicom_content:
            chest_nifti_files.append({
                "nifti_file": nifti_file,
                "labels": labels,
                "gcs_bucket_name": gcs_bucket_name,
                "gcs_images_dir": gcs_images_dir
            })

    return chest_nifti_files

def main():
    # Get all labels.
    grouped_labels_manager = GroupedLabelsManager(grouped_labels=GROUPED_GRADIENT_LABELS)
    all_labels = grouped_labels_manager.get_groups()
    print(f"All labels: {all_labels}")

    # Query data lake.
    client = bigquery.Client()
    query_job = client.query(DATALAKE_QUERY)
    results = query_job.result()

    # Populate training data.
    training_data = []
    for index, row in enumerate(results):
        primary_volumes = ast.literal_eval(row["primary_volumes"])
        structured_labels = ast.literal_eval(row["structured_labels"])
        gcs_bucket_name = row["gcs_bucket_name"]
        gcs_images_dir = row["gcs_images_dir"]

        labels = [key for key, value in structured_labels.items() if len(value) > 0]
        if "No Findings" in labels:
            labels.remove("No Findings")
        if "Other" in labels:
            labels.remove("Other")
        labels = [label.upper() for label in labels]
        multi_hot_encoding = [1 if item in labels else 0 for item in all_labels]

        if all(item == 1 for item in multi_hot_encoding):
            raise ValueError("All labels are 1")

        training_data.append({
            "nifti_files": primary_volumes,
            "labels": multi_hot_encoding,
            "gcs_bucket_name": gcs_bucket_name,
            "gcs_images_dir": gcs_images_dir
        })

    # Make sure all the volumes have 'Chest' body part in DICOM.
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_row, [row for row in training_data]), total=len(training_data), desc="Processing"))

    # Flatten results.
    training_data = [item for result in results for item in result]

    # Convert to pandas.
    out_df = pd.DataFrame(columns=["volume", "label"])
    for item in tqdm(training_data, total=len(training_data)):
        nifti_file = item["nifti_file"]
        labels = item["labels"]
        gcs_images_dir = item["gcs_images_dir"]

        new_row = {"volume": {"nifti_file": os.path.join(gcs_images_dir, nifti_file)}, "label": labels}
        out_df = out_df._append(new_row, ignore_index=True)

    # Save training data.
    out_df.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
