import ast
import os
from io import BytesIO, StringIO

import pandas as pd
from tqdm import tqdm

from epsutils.dicom import dicom_utils
from epsutils.gcs import gcs_utils
from epsutils.image import image_utils

GCS_REPORTS_FILE = "gs://epsilonlabs-filestore/cleaned_CRs/GRADIENT_CR_batch_1.csv"


def row_handler(row, index):
    report_text = row["report_text"]
    if "LUNG LESION" not in report_text.upper():
        return

    labels = [label.strip() for label in row["labels"].split(",")]

    image_paths_dict = ast.literal_eval(row["image_paths"])
    image_paths = []
    for value in image_paths_dict.values():
        image_paths.extend(value["paths"])

    for image_path in image_paths:
        image_path = os.path.join("gs://epsilon-data-us-central1/GRADIENT-DATABASE/CR/22JUL2024", image_path)
        gcs_data = gcs_utils.split_gcs_uri(image_path)
        content = gcs_utils.download_file_as_bytes(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])
        image = dicom_utils.get_dicom_image(BytesIO(content), custom_windowing_parameters={"window_center": 0, "window_width": 0})
        image = image_utils.numpy_array_to_pil_image(image)
        image.save(f"{os.path.basename(image_path)}.png")

    print("------------------------------------------")
    print("Report text:")
    print(report_text)
    print("Labels:")
    print(labels)
    key = input("Press any key to continue...")


def main():
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
