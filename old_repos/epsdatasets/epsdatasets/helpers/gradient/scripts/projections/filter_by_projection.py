import logging
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from epsutils.gcs import gcs_utils
from epsutils.logging import logging_utils

EPSILON_GCS_BUCKET_NAME = "gradient-crs"
EPSILON_GCS_IMAGES_DIR = "22JUL2024"
CONTENT_TO_SEARCH = "(0008,103E) Series Description:"
OUTPUT_FILE = "gradient-crs-22JUL2024-frontal-views.csv"


def process_file(file):
    dicom_content = gcs_utils.download_file_as_string(gcs_bucket_name=EPSILON_GCS_BUCKET_NAME, gcs_file_name=file)

    if CONTENT_TO_SEARCH not in dicom_content:
        return

    lines = dicom_content.splitlines()
    for line in lines:
        if line.startswith(CONTENT_TO_SEARCH):
            series_description = line[len(CONTENT_TO_SEARCH):].strip()
            words = series_description.lower().split()

            if {"pa", "ap"}.intersection(words) and {"chest", "rib", "ribs"}.intersection(words) and not {"lateral", "lat"}.intersection(words):
                logging.info(f"{file};{series_description}")

            return

    raise ValueError(f"Content '{CONTENT_TO_SEARCH}' not found in {file}")

def main():
    logging_utils.configure_logger(logger_file_name=OUTPUT_FILE, show_logging_level=False)

    # Get a list of files in the Epsilon GCS bucket.
    print(f"Getting a list of files in the {EPSILON_GCS_BUCKET_NAME} GCS bucket")
    files_in_bucket = gcs_utils.list_files(gcs_bucket_name=EPSILON_GCS_BUCKET_NAME, gcs_dir=EPSILON_GCS_IMAGES_DIR)
    files_in_bucket = [file for file in files_in_bucket if file.endswith(".txt")]
    files_in_bucket = set(files_in_bucket)  # Sets have average-time complexity of O(1) for lookups. In contrast, lists have an average-time complexity of O(n).
    print(f"Total files found: {len(files_in_bucket)}")

    with ProcessPoolExecutor() as executor:  # Use default number of workers.
        results = list(tqdm(executor.map(process_file, [file for file in files_in_bucket]), total=len(files_in_bucket), desc="Processing"))


if __name__ == "__main__":
    main()
