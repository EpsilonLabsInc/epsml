import logging
import os
from concurrent.futures import ThreadPoolExecutor

from google.cloud import storage
from tqdm import tqdm

from epsutils.logging import logging_utils

BUCKET_NAME = "epsilonlabs-dicom-store-main"
SUBFOLDER = "422ca224-a9f2-4c64-bf7c-bb122ae2a7bb/"  # Ensure the trailing slash.
LOCAL_FOLDER = "/mnt/efs/all-cxr/simonmed/images/422ca224-a9f2-4c64-bf7c-bb122ae2a7bb"
SKIP_EXISTING_FILES = True
MAX_WORKERS = None


def download_blob(blob):
    if blob.name == SUBFOLDER:  # Ignore placeholder directories.
        return

    local_path = os.path.join(LOCAL_FOLDER, blob.name[len(SUBFOLDER):])  # Remove subfolder prefix.

    if os.path.exists(local_path) and SKIP_EXISTING_FILES:
        return

    os.makedirs(os.path.dirname(local_path), exist_ok=True)  # Ensure subdirectories exist.
    blob.download_to_filename(local_path)

    logging.info(f"Succesully downloaded {local_path}")

    return blob.name


def download_files_from_gcs(bucket_name, subfolder, local_folder):
    logging.info("Creating GCS client")

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    logging.info(f"Listing files in gs://{bucket_name}/{subfolder}")

    blobs = list(bucket.list_blobs(prefix=subfolder))

    logging.info(f"Found total {len(blobs)} files")

    if not os.path.exists(local_folder):
        os.makedirs(local_folder)

    logging.info("Download started")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = list(tqdm(executor.map(download_blob, blobs), total=len(blobs), desc="Downloading"))


if __name__ == "__main__":
    logging_utils.configure_logger(logger_file_name=f"download_files.log")

    download_files_from_gcs(BUCKET_NAME, SUBFOLDER, LOCAL_FOLDER)
