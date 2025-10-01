import logging
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from epsutils.gcs import gcs_utils
from epsutils.logging import logging_utils

EPSILON_GCS_BUCKET_NAME = "gradient-cts-nifti"
EPSILON_GCS_IMAGES_DIR = "16AGO2024"
CONTENT_TO_SEARCH = "(0018,0015) Body Part Examined:"
LOG_IF_FOUND = True
LOG_IF_NOT_FOUND = True
OUTPUT_FILE = "scan_results.txt"


def process_file(file):
    dicom_content = gcs_utils.download_file_as_string(gcs_bucket_name=EPSILON_GCS_BUCKET_NAME, gcs_file_name=file)

    if CONTENT_TO_SEARCH in dicom_content:
        if LOG_IF_FOUND:
            logging.info(f"True ({file})")
    else:
        if LOG_IF_NOT_FOUND:
            logging.info(f"False ({file})")


def main():
    logging_utils.configure_logger(logger_file_name=OUTPUT_FILE)

    # Get a list of files in the Epsilon GCS bucket.
    print("Getting a list of files in the Epsilon GCS bucket")
    files_in_bucket = gcs_utils.list_files(gcs_bucket_name=EPSILON_GCS_BUCKET_NAME, gcs_dir=EPSILON_GCS_IMAGES_DIR)
    files_in_bucket = [file for file in files_in_bucket if file.endswith(".txt")]
    files_in_bucket = set(files_in_bucket)  # Sets have average-time complexity of O(1) for lookups. In contrast, lists have an average-time complexity of O(n).
    print(f"Total files found: {len(files_in_bucket)}")

    with ProcessPoolExecutor() as executor:  # Use default number of workers.
        results = list(tqdm(executor.map(process_file, [file for file in files_in_bucket]), total=len(files_in_bucket), desc="Processing"))


if __name__ == "__main__":
    main()
