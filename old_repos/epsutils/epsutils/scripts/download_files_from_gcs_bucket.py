import os
from concurrent.futures import ThreadPoolExecutor

from google.cloud import storage
from tqdm import tqdm

BUCKET_NAME = "epsilonlabs-dicom-store-main"
SUBFOLDER = "422ca224-a9f2-4c64-bf7c-bb122ae2a7bb/"  # Ensure the trailing slash.
LOCAL_FOLDER = "/mnt/efs/all-cxr/simonmed/batch1/422ca224-a9f2-4c64-bf7c-bb122ae2a7bb"
MAX_WORKERS = 8


def download_blob(blob):
    if blob.name == SUBFOLDER:  # Ignore placeholder directories.
        return

    local_path = os.path.join(LOCAL_FOLDER, blob.name[len(SUBFOLDER):])  # Remove subfolder prefix.
    os.makedirs(os.path.dirname(local_path), exist_ok=True)  # Ensure subdirectories exist.
    blob.download_to_filename(local_path)
    return blob.name


def download_files_from_gcs(bucket_name, subfolder, local_folder):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=subfolder))

    if not os.path.exists(local_folder):
        os.makedirs(local_folder)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = list(tqdm(executor.map(download_blob, blobs), total=len(blobs), desc="Downloading"))


if __name__ == "__main__":
    download_files_from_gcs(BUCKET_NAME, SUBFOLDER, LOCAL_FOLDER)
