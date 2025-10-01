import logging
import os
from concurrent.futures import ProcessPoolExecutor

import dicom2nifti.settings as settings
from google.cloud import storage
from tqdm import tqdm

from epsutils.logging import logging_utils

import config
from process_nifti_file import process_nifti_file


def main():
    # Configure logger.
    logging_utils.configure_logger(logger_file_name=os.path.join(config.OUTPUT_DIR, config.DISPLAY_NAME + "-log.txt"),
                                   logging_level=logging.WARNING)

    # Show configuration settings.
    config.dump_config()

    # Get all the files in the bucket.
    client = storage.Client()
    bucket = client.bucket(config.EPSILON_SOURCE_GCS_BUCKET_NAME)
    print(f"Getting all the files in the {config.EPSILON_SOURCE_GCS_BUCKET_NAME} GCS bucket")
    blobs = bucket.list_blobs(prefix=config.EPSILON_SOURCE_GCS_IMAGES_DIR)

    # Select NIfTI files only.
    print(f"Selecting NIfTI files")
    nifti_files = [os.path.basename(blob.name) for blob in blobs if blob.name.endswith(".nii.gz")]
    print(f"Num NIfTI files found: {len(nifti_files)}")

    # Supress dicom2nifti errors.
    settings.disable_validate_slice_increment()

    # Process in parallel.
    with ProcessPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(process_nifti_file, nifti_files), total=len(nifti_files), desc="Processing"))


if __name__ == "__main__":
    main()
