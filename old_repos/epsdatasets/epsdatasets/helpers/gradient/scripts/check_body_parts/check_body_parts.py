import logging
import os

from google.cloud import storage

from epsutils.logging import logging_utils

import config
import tasks


def main():
    # Configure logger.
    logging_utils.configure_logger(logger_file_name=os.path.join(config.OUTPUT_DIR, config.DISPLAY_NAME + "-log.txt"),
                                   logging_level=logging.INFO,
                                   show_logging_level=False)

    # Show configuration settings.
    config.dump_config()

    # Get all the files in the bucket.
    client = storage.Client()
    bucket = client.bucket(config.GCS_BUCKET_NAME)
    print(f"Getting all the files in the {config.GCS_BUCKET_NAME} GCS bucket")
    blobs = bucket.list_blobs(prefix=config.GCS_IMAGES_DIR)

    # Select NIfTI files only.
    print(f"Selecting NIfTI files")
    nifti_files = [os.path.basename(blob.name) for blob in blobs if blob.name.endswith(".nii.gz")]
    print(f"Num NIfTI files found: {len(nifti_files)}")

    # Start body parts check.
    print("Starting body parts check")
    tasks.start(nifti_files=nifti_files)
    print("Finished")


if __name__ == "__main__":
    main()
