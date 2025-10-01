import logging
import os
from concurrent.futures import ProcessPoolExecutor
from enum import Enum

from tqdm import tqdm

from epsutils.gcs import gcs_utils
from epsutils.logging import logging_utils

class Projection(Enum):
    FRONTAL = 1
    LATERAL = 2

EPSILON_GCS_BUCKET_NAME = "gradient-crs"
EPSILON_GCS_IMAGES_DIR = "22JUL2024"
GRADIENT_GCS_IMAGES_DIR = "GRADIENT-DATABASE/CR/22JUL2024"
CONTENT_TO_SEARCH = "(0008,103E) Series Description:"
PROJECTION = Projection.FRONTAL
OUTPUT_FILE = "gradient-crs-22JUL2024-frontal-views.csv"


def process_file(file):
    try:
        dicom_content = gcs_utils.download_file_as_string(gcs_bucket_name=EPSILON_GCS_BUCKET_NAME, gcs_file_name=file)
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return

    if CONTENT_TO_SEARCH not in dicom_content:
        return

    lines = dicom_content.splitlines()
    for line in lines:
        if line.startswith(CONTENT_TO_SEARCH):
            series_description = line[len(CONTENT_TO_SEARCH):].strip()
            words = series_description.lower().split()

            if PROJECTION == Projection.FRONTAL:
                if {"pa", "ap"}.intersection(words) and {"chest", "rib", "ribs"}.intersection(words) and not {"lateral", "lat"}.intersection(words):
                    gradient_file_name = os.path.join(GRADIENT_GCS_IMAGES_DIR, os.path.basename(file).replace("_", "/").replace(".txt", ".dcm"))
                    logging.info(f"{gradient_file_name};{series_description}")
            elif PROJECTION == Projection.LATERAL:
                if {"lat", "lateral"}.intersection(words) and {"chest", "rib", "ribs"}.intersection(words) and not {"pa", "ap"}.intersection(words):
                    gradient_file_name = os.path.join(GRADIENT_GCS_IMAGES_DIR, os.path.basename(file).replace("_", "/").replace(".txt", ".dcm"))
                    logging.info(f"{gradient_file_name};{series_description}")
            else:
                raise ValueError(f"Unsupported projection {PROJECTION}")

            return

    raise ValueError(f"Content '{CONTENT_TO_SEARCH}' not found in {file}")

def main():
    logging_utils.configure_logger(logger_file_name=OUTPUT_FILE, show_logging_level=False)

    # Get a list of files in the Epsilon GCS bucket.
    print(f"Getting a list of files in the '{EPSILON_GCS_BUCKET_NAME}' GCS bucket")
    files_in_bucket = gcs_utils.list_files(gcs_bucket_name=EPSILON_GCS_BUCKET_NAME, gcs_dir=EPSILON_GCS_IMAGES_DIR)
    files_in_bucket = [file for file in files_in_bucket if file.endswith(".txt")]
    files_in_bucket = set(files_in_bucket)  # Sets have average-time complexity of O(1) for lookups. In contrast, lists have an average-time complexity of O(n).
    print(f"Total files found: {len(files_in_bucket)}")

    with ProcessPoolExecutor() as executor:  # Use default number of workers.
        results = list(tqdm(executor.map(process_file, [file for file in files_in_bucket]), total=len(files_in_bucket), desc="Processing"))


if __name__ == "__main__":
    main()
