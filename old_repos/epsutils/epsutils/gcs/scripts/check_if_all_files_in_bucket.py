import argparse
import ast
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from epsutils.gcs import gcs_utils


def check_row(row_dict, base_path_gcs_substitutions, max_retries, retry_delay_in_sec):
    image_paths = ast.literal_eval(row_dict["relative_image_paths"])
    base_path = row_dict["base_path"]

    if base_path not in base_path_gcs_substitutions:
        print(f"Base path {base_path} not in base path GCS substitutions dict")
        return

    gcs_data = gcs_utils.split_gcs_uri(base_path_gcs_substitutions[base_path])
    gcs_image_paths = [os.path.join(gcs_data["gcs_path"], image_path) for image_path in image_paths]

    attempt = 1
    while True:
        try:
            if not gcs_utils.check_if_files_exist(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_names=gcs_image_paths):
                print(f"One or more items in {gcs_image_paths} not found in GCS")
            return
        except Exception as e:
            print(f"GCS check failed on attempt {attempt}: {e}")
            if max_retries is not None and attempt >= max_retries:
                print("Max retries reached, skipping row")
                return
            print(f"Retrying in {retry_delay_in_sec} seconds...")
            time.sleep(retry_delay_in_sec)
            attempt += 1


def main(args):
    print(f"Loading {args.input_csv_reports_file}")
    df = pd.read_csv(args.input_csv_reports_file, low_memory=False)

    if args.total_slices is not None and args.slice_index is not None:
        print(f"Dataset length before slicing: {len(df)}")
        slices = np.array_split(df, args.total_slices)
        print(f"Extracting slice {args.slice_index}")
        df = slices[args.slice_index]
        print(f"Slice length: {len(df)}")

    print("Converting dataset to a list of dicts")
    row_dicts = [row.to_dict() for _, row in df.iterrows()]

    print("Scanning dataset for files and checking their presence in GCS")
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(tqdm(executor.map(lambda row: check_row(row, args.base_path_gcs_substitutions, args.max_retries, args.retry_delay_in_sec), row_dicts), total=len(row_dicts), desc="Processing"))


if __name__ == "__main__":
    INPUT_CSV_REPORTS_FILE = "./gradient_batches_1-6_segmed_batches_1-16_simonmed_batches_1-10_reports_with_labels_all.csv"
    TOTAL_SLICES = 13  # Use None for no slicing.
    SLICE_INDEX = 0  # Zero-based. Use None for no slicing.
    MAX_RETRIES = None  # If None, retry indefinitely.
    RETRY_DELAY_IN_SEC = 2

    BASE_PATH_GCS_SUBSTITUTIONS = {
            "gradient_09JAN2025": "gs://epsilonlabs-datasets/dicom/gradient/09JAN2025",
            "gradient_13JAN2025": "gs://epsilonlabs-datasets/dicom/gradient/13JAN2025/deid",
            "gradient_16AUG2024": "gs://epsilonlabs-datasets/dicom/gradient/16AUG2024",
            "gradient_20DEC2024": "gs://epsilonlabs-datasets/dicom/gradient/20DEC2024",
            "gradient_22JUL2024": "gs://epsilonlabs-datasets/dicom/gradient/22JUL2024",
            "gradient_01JUL2025": "gs://epsilonlabs-datasets/dicom/gradient-new/01JUL2025",
            "segmed_batch_1": "gs://epsilonlabs-datasets/dicom/segmed/batch1",
            "segmed_batch_2": "gs://epsilonlabs-datasets/dicom/segmed/batch2",
            "segmed_batch_3": "gs://epsilonlabs-datasets/dicom/segmed/batch3",
            "segmed_batch_4": "gs://epsilonlabs-datasets/dicom/segmed/batch4",
            "segmed_batch_5": "gs://epsilonlabs-datasets/dicom/segmed/batch5",
            "segmed_batch_6": "gs://epsilonlabs-datasets/dicom/segmed/batch6",
            "segmed_batch_7": "gs://epsilonlabs-datasets/dicom/segmed/batch7",
            "segmed_batch_8": "gs://epsilonlabs-datasets/dicom/segmed/batch8",
            "segmed_batch_9": "gs://epsilonlabs-datasets/dicom/segmed/batch9",
            "segmed_batch_10": "gs://epsilonlabs-datasets/dicom/segmed/batch10",
            "segmed_batch_11": "gs://epsilonlabs-datasets/dicom/segmed/batch11",
            "segmed_batch_12": "gs://epsilonlabs-datasets/dicom/segmed/batch12",
            "segmed_batch_13": "gs://epsilonlabs-datasets/dicom/segmed/batch13",
            "segmed_batch_14": "gs://epsilonlabs-datasets/dicom/segmed/batch14",
            "segmed_batch_15": "gs://epsilonlabs-datasets/dicom/segmed/batch15",
            "segmed_batch_16": "gs://epsilonlabs-datasets/dicom/segmed/batch16",
            "simonmed": "gs://epsilonlabs-datasets/dicom/simonmed"
    }

    args = argparse.Namespace(input_csv_reports_file=INPUT_CSV_REPORTS_FILE,
                              total_slices=TOTAL_SLICES,
                              slice_index=SLICE_INDEX,
                              max_retries=MAX_RETRIES,
                              retry_delay_in_sec=RETRY_DELAY_IN_SEC,
                              base_path_gcs_substitutions=BASE_PATH_GCS_SUBSTITUTIONS)

    main(args)
