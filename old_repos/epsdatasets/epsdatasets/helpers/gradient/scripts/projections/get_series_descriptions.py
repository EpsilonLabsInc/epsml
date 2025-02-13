import logging
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from epsutils.gcs import gcs_utils
from epsutils.logging import logging_utils

EPSILON_GCS_IMAGES_DIR = "gs://gradient-crs/22JUL2024"
GRADIENT_IMAGES_DIR = "GRADIENT-DATABASE/CR/22JUL2024"
CONTENT_TO_SEARCH = "(0008,103E) Series Description:"
OUTPUT_FILE = "gradient-crs-22JUL2024-series-descriptions.csv"


def process_file(file):
    gcs_data = gcs_utils.split_gcs_uri(EPSILON_GCS_IMAGES_DIR)
    dicom_content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=file)

    if CONTENT_TO_SEARCH not in dicom_content:
        return

    lines = dicom_content.splitlines()
    for line in lines:
        if not line.startswith(CONTENT_TO_SEARCH):
            continue

        series_description = line[len(CONTENT_TO_SEARCH):].strip()
        dicom_file = os.path.join(GRADIENT_IMAGES_DIR, os.path.basename(file).replace("_", "/").replace(".txt", ".dcm"))
        logging.info(f"{series_description};{dicom_file}")

    raise ValueError(f"Content '{CONTENT_TO_SEARCH}' not found in {file}")


def main():
    logging_utils.configure_logger(logger_file_name=OUTPUT_FILE, show_logging_level=False)

    print(f"Getting a list of TXT files in {EPSILON_GCS_IMAGES_DIR}")
    gcs_data = gcs_utils.split_gcs_uri(EPSILON_GCS_IMAGES_DIR)
    files_in_bucket = gcs_utils.list_files(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_dir=gcs_data["gcs_path"])
    files_in_bucket = [file for file in files_in_bucket if file.endswith(".txt")]
    files_in_bucket = set(files_in_bucket)  # Sets have average-time complexity of O(1) for lookups. In contrast, lists have an average-time complexity of O(n).
    print(f"Total TXT files found: {len(files_in_bucket)}")

    with ProcessPoolExecutor() as executor:  # Use default number of workers.
        results = list(tqdm(executor.map(process_file, [file for file in files_in_bucket]), total=len(files_in_bucket), desc="Processing"))


if __name__ == "__main__":
    main()
