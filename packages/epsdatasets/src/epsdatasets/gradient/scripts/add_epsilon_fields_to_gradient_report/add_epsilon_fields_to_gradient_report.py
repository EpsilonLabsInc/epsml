import ast
import os
from io import StringIO

import pandas as pd
from tqdm import tqdm

from epsdatasets.helpers.gradient import gradient_utils
from epsutils.gcs import gcs_utils

import config

def main():
    print("Using the following config:")
    print("-------------------------------------------")
    print(f"GRADIENT_GCS_BUCKET_NAME = {config.GRADIENT_GCS_BUCKET_NAME}")
    print(f"GRADIENT_GCS_REPORTS_FILE = {config.GRADIENT_GCS_REPORTS_FILE}")
    print(f"EPSILON_GCS_BUCKET_NAME = {config.EPSILON_GCS_BUCKET_NAME}")
    print(f"EPSILON_GCS_IMAGES_DIR = {config.EPSILON_GCS_IMAGES_DIR}")
    print(f"EPSILON_LOCAL_IMAGES_DIR = {config.EPSILON_LOCAL_IMAGES_DIR}")
    print(f"DESTINATION_GCS_BUCKET_NAME = {config.DESTINATION_GCS_BUCKET_NAME}")
    print(f"DESTINATION_GCS_REPORTS_FILE = {config.DESTINATION_GCS_REPORTS_FILE}")
    print("-------------------------------------------")

    # Download reports file.
    print(f"Downloading reports file from the GCS bucket")
    reports_file_content = gcs_utils.download_file_as_string(gcs_bucket_name=config.GRADIENT_GCS_BUCKET_NAME, gcs_file_name=config.GRADIENT_GCS_REPORTS_FILE)

    # Read reports file.
    print("Reading reports file")
    df = pd.read_csv(StringIO(reports_file_content), sep=",", low_memory=False)

    # Get a list of files in the Epsilon GCS bucket.
    print("Getting a list of files in the Epsilon GCS bucket")
    files_in_bucket = gcs_utils.list_files(gcs_bucket_name=config.EPSILON_GCS_BUCKET_NAME, gcs_dir=config.EPSILON_GCS_IMAGES_DIR)
    files_in_bucket = set(files_in_bucket)  # Sets have average-time complexity of O(1) for lookups. In contrast, lists have an average-time complexity of O(n).
    print(f"Total files found: {len(files_in_bucket)}")

    # Add new fields.
    print("Updating new fields.")
    df["StudyAccepted"] = False
    df["PrimaryVolumes"] = [[] for _ in range(len(df))]
    df["ContrastAgentVolumes"] = [[] for _ in range(len(df))]
    df["NonContrastAgentVolumes"] = [[] for _ in range(len(df))]
    df["GcsBucketName"] = config.EPSILON_GCS_BUCKET_NAME
    df["GcsImagesDir"] = config.EPSILON_GCS_IMAGES_DIR

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        patient_id = row["PatientID"]
        accession_number = row["AccessionNumber"]
        study_instance_uid = row["StudyInstanceUid"]
        series_instance_uids = ast.literal_eval(row["SeriesInstanceUid"])

        for series_instance_uid in series_instance_uids:
            nifti_file_name = gradient_utils.get_nifti_file_name(
                patient_id=patient_id, accession_number=accession_number, study_instance_uid=study_instance_uid, series_instance_uid=series_instance_uid)
            nifti_file_path = os.path.join(config.EPSILON_GCS_IMAGES_DIR, nifti_file_name)

            # Check if corresponding NIfTI file exists in the bucket.
            if not nifti_file_path in files_in_bucket:
                continue

            df.at[index, "StudyAccepted"] = True
            df.at[index, "PrimaryVolumes"].append(nifti_file_name)

            # Read volume info file and determine whether volume uses contrast agent or not.
            volume_info_file_name = gradient_utils.get_volume_info_file_name(
                patient_id=patient_id, accession_number=accession_number, study_instance_uid=study_instance_uid, series_instance_uid=series_instance_uid)
            if config.EPSILON_LOCAL_IMAGES_DIR is None:
                volume_info_file_path = os.path.join(config.EPSILON_GCS_IMAGES_DIR, volume_info_file_name)
                volume_info_content = gcs_utils.download_file_as_string(gcs_bucket_name=config.EPSILON_GCS_BUCKET_NAME, gcs_file_name=volume_info_file_path)
            else:
                volume_info_file_path = os.path.join(config.EPSILON_LOCAL_IMAGES_DIR, volume_info_file_name)
                try:
                    with open(volume_info_file_path, "r") as file:
                        volume_info_content = file.read()
                except Exception as e:
                    print(f"Error reading volume info file: {e}")
                    continue

            if "(0018,0010) Contrast/Bolus Agent" in volume_info_content:
                df.at[index, "ContrastAgentVolumes"].append(nifti_file_name)
            else:
                df.at[index, "NonContrastAgentVolumes"].append(nifti_file_name)

    # Upload updated reports file.
    print("Uploading updated reports file")
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    upload_data = [{"is_file": False, "local_file_or_string": csv_buffer.getvalue(), "gcs_file_name": config.DESTINATION_GCS_REPORTS_FILE}]
    gcs_utils.upload_files(upload_data=upload_data, gcs_bucket_name=config.DESTINATION_GCS_BUCKET_NAME)


if __name__ == "__main__":
    main()
