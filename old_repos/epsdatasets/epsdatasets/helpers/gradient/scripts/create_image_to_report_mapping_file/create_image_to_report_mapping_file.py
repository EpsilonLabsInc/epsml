import ast
import os
from io import StringIO

import pandas as pd
from tqdm import tqdm

from epsutils.gcs import gcs_utils

import config


def main():
    print("Using the following config:")
    print("-------------------------------------------")
    print(f"GRADIENT_GCS_BUCKET_NAME = {config.GRADIENT_GCS_BUCKET_NAME}")
    print(f"GRADIENT_GCS_REPORTS_FILE = {config.GRADIENT_GCS_REPORTS_FILE}")
    print(f"GRADIENT_GCS_ROOT_IMAGES_DIR = {config.GRADIENT_GCS_ROOT_IMAGES_DIR}")
    print(f"EPSILON_GCS_BUCKET_NAME = {config.EPSILON_GCS_BUCKET_NAME}")
    print(f"EPSILON_GCS_IMAGES_DIR = {config.EPSILON_GCS_IMAGES_DIR}")
    print(f"OUT_FILE_NAME = {config.OUT_FILE_NAME}")
    print("-------------------------------------------")

    # Download reports file from the Gradient GCS bucket.
    print(f"Downloading reports file from the Gradient GCS bucket")
    reports_file_content = gcs_utils.download_file_as_string(gcs_bucket_name=config.GRADIENT_GCS_BUCKET_NAME, gcs_file_name=config.GRADIENT_GCS_REPORTS_FILE)

    # Read reports file.
    print("Reading reports file")
    in_df = pd.read_csv(StringIO(reports_file_content), sep=",", low_memory=False)

    # Get a list of file paths in the Epsilon GCS bucket.
    print("Getting a list of files in the Epsilon GCS bucket")
    files_in_bucket = gcs_utils.list_files(gcs_bucket_name=config.EPSILON_GCS_BUCKET_NAME, gcs_dir=config.EPSILON_GCS_IMAGES_DIR)
    files_in_bucket = set(files_in_bucket)  # Sets have average-time complexity of O(1) for lookups. In contrast, lists have an average-time complexity of O(n).
    print(f"Total files found: {len(files_in_bucket)}")

    # Create a dict of row IDs.
    print("Creating a dict of row IDs")
    row_ids = {}
    for _, row in tqdm(in_df.iterrows(), total=len(in_df), desc="Processing"):
        row_id = row["row_id"]
        patient_id = row["PatientID"]
        accession_number = row["AccessionNumber"]
        study_instance_uid = row["StudyInstanceUid"]
        try:
            series_instance_uids = ast.literal_eval(row["SeriesInstanceUid"])
        except Exception as e:
            series_instance_uids= [row["SeriesInstanceUid"]]

        for series_instance_uid in series_instance_uids:
            prefix = patient_id + "_" + accession_number + "_studies_" + study_instance_uid + "_series_" + series_instance_uid + "_instances_"
            prefix = os.path.join(config.EPSILON_GCS_IMAGES_DIR, prefix)
            row_ids[prefix] = row_id

    # Create out table.
    out_df = pd.DataFrame(columns=["ImagePath", "RowId"])

    # Fill the out table by assigning row IDs to the image files.
    print("Assigning row IDs to image files")
    pattern = "_instances_"
    for file_name in tqdm(files_in_bucket, total=len(files_in_bucket), desc="Processing"):
        pos = file_name.find(pattern)
        if pos == -1:
            print(f"Cannot locate '{pattern}' in '{file_name}'")
            continue

        prefix = file_name[:pos + len(pattern)]
        assert prefix in row_ids
        row_id = row_ids[prefix]

        dicom_file = file_name.replace("_", "/").replace(".txt", ".dcm")
        new_row = {"ImagePath": os.path.join(config.GRADIENT_GCS_ROOT_IMAGES_DIR, dicom_file), "RowId": row_id}
        out_df = out_df._append(new_row, ignore_index=True)

    # Save out table as CSV file.
    out_df.to_csv(config.OUT_FILE_NAME, index=False)


if __name__ == "__main__":
    main()
