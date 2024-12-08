import logging
import os

from google.cloud import storage

from epsutils.logging import logging_utils

import config
import tasks


def main():
    # Configure logger.
    logger_file_name = os.path.join(config.OUTPUT_DIR, config.DISPLAY_NAME + "-log.txt")
    logging_utils.configure_logger(logger_file_name=logger_file_name,
                                   logging_level=logging.INFO,
                                   show_logging_level=False,
                                   append_to_existing_log_file=config.SKIP_ALREADY_PROCESSED_IMAGES)

    # Show configuration settings.
    config.dump_config()

    # Get a list of already processed images.
    if config.SKIP_ALREADY_PROCESSED_IMAGES:
        print("Getting a list of already processed NIfTI files")
        nifti_files_to_skip = []

        with open(logger_file_name, "r") as file:
            for line in file:
                start_index = line.find(config.GCS_IMAGES_DIR + "/")
                if start_index != -1:
                    nifti_file_name = os.path.basename(line[start_index:].replace(")", "").strip())
                    nifti_files_to_skip.append(nifti_file_name)

        print(f"{len(nifti_files_to_skip)} NIfTI files were already processed and will be skipped")

        # Sets offer average O(1) time complexity for membership checks, making them much faster than lists.
        nifti_files_to_skip = set(nifti_files_to_skip)

    # Get all the files in the bucket.
    client = storage.Client()
    bucket = client.bucket(config.GCS_BUCKET_NAME)
    print(f"Getting all the files in the {config.GCS_BUCKET_NAME} GCS bucket")
    blobs = bucket.list_blobs(prefix=config.GCS_IMAGES_DIR)

    # Select NIfTI files only.
    print(f"Selecting NIfTI files")
    nifti_files = [os.path.basename(blob.name) for blob in blobs if blob.name.endswith(".nii.gz")]
    print(f"Num NIfTI files found: {len(nifti_files)}")

    # Remove already processed images.
    if config.SKIP_ALREADY_PROCESSED_IMAGES:
        nifti_files = [f for f in nifti_files if f not in nifti_files_to_skip]
        print(f"Only {len(nifti_files)} NIfTI files will be processed, already processed files will be skipped")

    # Start body parts check.
    print("Starting body parts check")
    tasks.start(nifti_files=nifti_files)
    print("Finished")


if __name__ == "__main__":
    main()
