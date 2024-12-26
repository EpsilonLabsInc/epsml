import logging
import os
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO

import pydicom
from tqdm import tqdm

from epsutils.dicom import dicom_utils
from epsutils.gcs import gcs_utils
from epsutils.logging import logging_utils

OUTPUT_FILE = "./output/gradient_crs_22JUL2024_corrupt_dicoms.txt"
EPSILON_GCS_BUCKET_NAME = "gradient-crs"
EPSILON_GCS_DIR = "22JUL2024"
GRADIENT_GCS_BUCKET_NAME = "epsilon-data-us-central1"
GRADIENT_GCS_IMAGES_DIR = "GRADIENT-DATABASE/CR/22JUL2024"


def check_dicom_file(txt_file_name):
    dicom_file_name = txt_file_name.replace("_", "/").replace(".txt", ".dcm")
    dicom_file_name = os.path.join(GRADIENT_GCS_IMAGES_DIR, dicom_file_name)

    try:
        content = gcs_utils.download_file_as_bytes(gcs_bucket_name=GRADIENT_GCS_BUCKET_NAME, gcs_file_name=dicom_file_name)
        dataset = pydicom.dcmread(BytesIO(content))
        res = dicom_utils.check_dicom_image_in_dataset(dataset)
    except Exception as e:
        logging.error(f"{str(e)};{os.path.join(EPSILON_GCS_DIR, txt_file_name)}")


def main():
    logging_utils.configure_logger(logger_file_name=OUTPUT_FILE, show_logging_level=False)

    # Get all files in the bucket.
    print(f"Getting a list of all the files in '{EPSILON_GCS_DIR}' dir of the '{EPSILON_GCS_BUCKET_NAME}' GCS bucket")
    all_files = gcs_utils.list_files(gcs_bucket_name=EPSILON_GCS_BUCKET_NAME, gcs_dir=EPSILON_GCS_DIR)
    print(f"All files found: {len(all_files)}")

    # Extract only TXT files.
    txt_files = [os.path.basename(file) for file in all_files if file.endswith(".txt")]
    print(f"All TXT files found: {len(txt_files)}")

    # Check DICOM files in parallel.
    with ProcessPoolExecutor() as executor:  # Use default number of workers.
        results = list(tqdm(executor.map(check_dicom_file, txt_files), total=len(txt_files), desc="Checking"))


if __name__ == "__main__":
    main()
