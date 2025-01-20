import logging
import os
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from epsutils.gcs import gcs_utils
from epsutils.logging import logging_utils

GCS_IMAGES_DIR = "gs://gradient-crs/13JAN2025"
LIST_FILE_NAMES_ONLY = True
CONVERT_TXT_TO_DICOM_PATHS = True
OUTPUT_FILE = "./output/gradient-crs-13JAN2025.csv"


def list_file(file):
    if LIST_FILE_NAMES_ONLY:
        file_to_list = os.path.basename(file)

        if CONVERT_TXT_TO_DICOM_PATHS:
            file_to_list = file_to_list.replace("_", "/").replace(".txt", ".dcm")
    else:
        file_to_list = file

    logging.info(f"{file_to_list}")


def main():
    logging_utils.configure_logger(logger_file_name=OUTPUT_FILE, show_logging_level=False)

    print(f"Getting a list of TXT files in {GCS_IMAGES_DIR}")
    gcs_data = gcs_utils.split_gcs_uri(GCS_IMAGES_DIR)
    files_in_bucket = gcs_utils.list_files(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_dir=gcs_data["gcs_path"])
    files_in_bucket = [file for file in files_in_bucket if file.endswith(".txt")]
    files_in_bucket = set(files_in_bucket)  # Sets have average-time complexity of O(1) for lookups. In contrast, lists have an average-time complexity of O(n).
    print(f"Total files found: {len(files_in_bucket)}")

    with ProcessPoolExecutor() as executor:  # Use default number of workers.
        results = list(tqdm(executor.map(list_file, [file for file in files_in_bucket]), total=len(files_in_bucket), desc="Processing"))


if __name__ == "__main__":
    main()
