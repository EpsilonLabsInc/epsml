import logging
import os
from io import StringIO

import pandas as pd
from tqdm import tqdm

from epsutils.gcs import gcs_utils
from epsutils.logging import logging_utils

GRADIENT_REPORTS_FILE_GCS_URI = "gs://epsilon-data-us-central1/GRADIENT-DATABASE/CR/09JAN2025/reports_20250108153205.csv"
EPSILON_IMAGES_DIR_GCS_URI = "gs://gradient-crs/09JAN2025"
OUTPUT_FILE = "./output/gradient-crs-09JAN2025-rejected-studies.csv"


def main():
    logging_utils.configure_logger(logger_file_name=OUTPUT_FILE, show_logging_level=False)

    print(f"Downloading reports file {GRADIENT_REPORTS_FILE_GCS_URI}")
    gcs_data = gcs_utils.split_gcs_uri(GRADIENT_REPORTS_FILE_GCS_URI)
    content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])
    reports_dataset = pd.read_csv(StringIO(content), low_memory=False)
    print(f"Num studies: {len(reports_dataset)}")

    print(f"Getting a list of TXT files in {EPSILON_IMAGES_DIR_GCS_URI}")
    gcs_data = gcs_utils.split_gcs_uri(EPSILON_IMAGES_DIR_GCS_URI)
    files_in_bucket = gcs_utils.list_files(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_dir=gcs_data["gcs_path"])
    prefixes = ["_".join(os.path.basename(file).split("_")[:4]) for file in files_in_bucket if file.endswith(".txt")]
    prefixes = set(prefixes)  # Sets have average-time complexity of O(1) for lookups. In contrast, lists have an average-time complexity of O(n).
    print(f"Total prefixes found: {len(prefixes)}")

    for index, row in tqdm(reports_dataset.iterrows(), total=len(reports_dataset), desc="Processing"):
        patient_id = row["PatientID"]
        accession_number = row["AccessionNumber"]
        study_instance_uid = row["StudyInstanceUid"]
        prefix = f"{patient_id}_{accession_number}_studies_{study_instance_uid}"

        if prefix not in prefixes:
            logging.info(f"{prefix}")


if __name__ == "__main__":
    main()
