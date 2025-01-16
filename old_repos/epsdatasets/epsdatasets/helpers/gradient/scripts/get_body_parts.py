import logging
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from epsutils.gcs import gcs_utils
from epsutils.logging import logging_utils

EPSILON_GCS_BUCKET_NAME = "gradient-crs"
EPSILON_GCS_IMAGES_DIR = "16AG02924"
CONTENT_TO_SEARCH = "(0018,0015) Body Part Examined:"
OUTPUT_FILE = "./output/gradient-crs-16AG02924-body-parts.txt"


def process_file(file):
    dicom_content = gcs_utils.download_file_as_string(gcs_bucket_name=EPSILON_GCS_BUCKET_NAME, gcs_file_name=file)

    lines = dicom_content.splitlines()
    for line in lines:
        if line.startswith(CONTENT_TO_SEARCH):
            body_part = line[len(CONTENT_TO_SEARCH):].strip()
            logging.info(f"{file};{body_part}")
            return

    raise ValueError(f"Content {CONTENT_TO_SEARCH} not found in {file}")


def main():
    logging_utils.configure_logger(logger_file_name=OUTPUT_FILE, show_logging_level=False)

    print(f"Getting a list of TXT files in gs://{EPSILON_GCS_BUCKET_NAME}/{EPSILON_GCS_IMAGES_DIR}")
    files_in_bucket = gcs_utils.list_files(gcs_bucket_name=EPSILON_GCS_BUCKET_NAME, gcs_dir=EPSILON_GCS_IMAGES_DIR)
    files_in_bucket = [file for file in files_in_bucket if file.endswith(".txt")]
    files_in_bucket = set(files_in_bucket)  # Sets have average-time complexity of O(1) for lookups. In contrast, lists have an average-time complexity of O(n).
    print(f"Total TXT files found: {len(files_in_bucket)}")

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, [file for file in files_in_bucket]), total=len(files_in_bucket), desc="Processing"))


if __name__ == "__main__":
    main()
