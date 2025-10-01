import logging
import os
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
from tqdm import tqdm

from epsutils.gcs import gcs_utils
from epsutils.logging import logging_utils

EPSILON_GCS_BUCKET_NAME = "gradient-cts-fixed-nifti"
EPSILON_GCS_IMAGES_DIR = "16AGO2024"
OUTPUT_FILE_NAME = "/home/andrej/data/gradient-cts-fixed-nifti-16AGO2024-slices-consistency.txt"


def worker(nifti_file):
    try:
        # Download txt file.
        txt_file = nifti_file.replace(".nii.gz", ".txt")
        content = gcs_utils.download_file_as_string(gcs_bucket_name=EPSILON_GCS_BUCKET_NAME, gcs_file_name=txt_file)

        # Read number of slices and NIfTI generator.
        lines = content.split("\n")
        num_slices_txt = None
        nifti_generator = None
        for line in lines:
            if line.startswith("Number of slices:"):
                value = line[len("Number of slices:") + 1:]
                num_slices_txt = int(value.strip())

            if line.startswith("NIfTI generator:"):
                nifti_generator = line

            if num_slices_txt is not None and nifti_generator is not None:
                break

        if num_slices_txt is None:
            raise RuntimeError("Error reading number of slices")

        # Download NIfTI file.
        local_nifti_file = os.path.basename(nifti_file)
        gcs_utils.download_file(gcs_bucket_name=EPSILON_GCS_BUCKET_NAME, gcs_file_name=nifti_file, local_file_name=local_nifti_file)
        if not os.path.exists(local_nifti_file):
            raise RuntimeError("File not properly downloaded")

        # Get number of slices.
        image = nib.load(local_nifti_file)
        img_data = image.get_fdata()
        num_slices_nifti = img_data.shape[2]

        if num_slices_txt != num_slices_nifti:
            logging.error(f"Different number of slices. TXT = {num_slices_txt}, NIfTI = {num_slices_nifti} ({nifti_file}, {nifti_generator})")

    except Exception as e:
        logging.error(f"{str(e)} ({nifti_file})")
    finally:
        if os.path.exists(local_nifti_file):
            os.remove(local_nifti_file)


def main():
    logging_utils.configure_logger(logger_file_name=OUTPUT_FILE_NAME)

    print(f"Getting all the files in the {EPSILON_GCS_BUCKET_NAME} GCS bucket")
    files = gcs_utils.list_files(gcs_bucket_name=EPSILON_GCS_BUCKET_NAME, gcs_dir=EPSILON_GCS_IMAGES_DIR)
    print(f"Num all files: {len(files)}")

    print(f"Gathering NIfTI files")
    nifti_files = [file for file in files if file.endswith(".nii.gz")]
    print(f"Num all NIfTI files: {len(nifti_files)}")

    with ProcessPoolExecutor() as executor:  # Use default number of workers.
            results = list(tqdm(executor.map(worker, nifti_files), total=len(nifti_files), desc="Processing"))


if __name__ == "__main__":
    main()
