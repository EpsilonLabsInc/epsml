import ast
import os
import logging
from io import StringIO

import pandas as pd
from tqdm import tqdm

from epsutils.gcs import gcs_utils
from epsutils.logging import logging_utils

GCS_REPORTS_FILE = "gs://epsilonlabs-filestore/cleaned_CRs/GRADIENT_CR_batch_1_chest_with_image_paths.csv"
GCS_IMAGES_DIR = "GRADIENT-DATABASE/CR/22JUL2024"
OUTPUT_FILE = "gradient-crs-22JUL2024-adults-only.csv"


def get_age(age_str):
    if not isinstance(age_str, str):
        return None

    if not age_str.upper().endswith("Y"):
        return None

    try:
        age = int(age_str[:-1])
    except:
        return None

    return age


def row_handler(row, index):
    age = get_age(row["age"])

    if age is None or age < 18:
        return

    try:
        image_paths_dict = ast.literal_eval(row["image_paths"])
    except:
        return

    image_paths = []
    for value in image_paths_dict.values():
        image_paths.extend(value["paths"])

    for image_path in image_paths:
        logging.info(f"{age};{os.path.join(GCS_IMAGES_DIR, image_path)}")


def main():
    logging_utils.configure_logger(logger_file_name=OUTPUT_FILE, show_logging_level=False)

    print(f"Downloading reports file {GCS_REPORTS_FILE}")
    gcs_data = gcs_utils.split_gcs_uri(GCS_REPORTS_FILE)
    content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

    print("Loading reports file")
    df = pd.read_csv(StringIO(content))

    print("Reading reports file")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        row_handler(row, index)


if __name__ == "__main__":
    main()
