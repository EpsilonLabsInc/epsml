import argparse
import ast
import os

import pandas as pd
from tqdm import tqdm

from epsutils.gcs import gcs_utils


def main(args):
    print(f"Loading {args.input_csv_reports_file}")
    df = pd.read_csv(args.input_csv_reports_file, low_memory=False)

    print("Scanning dataset for files and checking their presence in GCS")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        image_paths = ast.literal_eval(row["relative_image_paths"])
        base_path = row["base_path"]

        if base_path not in args.base_path_gcs_substitutions:
            raise ValueError(f"Base path {base_path} not in base path GCS substitutions dict")

        gcs_data = gcs_utils.split_gcs_uri(args.base_path_gcs_substitutions[base_path])
        gcs_image_paths = [os.path.join(gcs_data["gcs_path"], image_path) for image_path in image_paths]

        if not gcs_utils.check_if_files_exist(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_names=gcs_image_paths):
            print(f"One or more items in {gcs_image_paths} not found in GCS")


if __name__ == "__main__":
    INPUT_CSV_REPORTS_FILE = "./gradient_batches_1-6_segmed_batches_1-16_simonmed_batches_1-10_reports_with_labels_all.csv"

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
                              base_path_gcs_substitutions=BASE_PATH_GCS_SUBSTITUTIONS)

    main(args)
