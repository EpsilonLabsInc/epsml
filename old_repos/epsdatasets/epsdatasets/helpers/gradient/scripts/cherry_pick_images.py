import ast
import os
from io import StringIO

import pandas as pd
from tqdm import tqdm

from epsutils.dicom import dicom_utils
from epsutils.gcs import gcs_utils
from epsutils.image import image_utils

GCS_INPUT_FILE = "gs://report_csvs/cleaned/CR/labels_for_binary_classification/GRADIENT_CR_ALL_CHEST_BATCHES_cleaned_pleura_labels.csv"
GCS_INPUT_IMAGES_DIR = "GRADIENT-DATABASE/CR"
GCS_FRONTAL_PROJECTIONS_FILE = "gs://gradient-crs/archive/training/projections/gradient-crs-all-batches-chest-only-frontal-projections.csv"
GCS_LATERAL_PROJECTIONS_FILE = "gs://gradient-crs/archive/training/projections/gradient-crs-all-batches-chest-only-lateral-projections.csv"
LABELS_COLUMN_NAME = "pleura_labels"
TARGET_BATCH_ID = "22JUL2024"
TARGET_LABEL_NAME = "Pleural Effusion"
TARGET_IMAGES_DIR = "/mnt/efs/all-cxr/gradient"
TARGET_NUM_IMAGES = 250
DESTINATION_DIR = "/home/andrej/tmp/pleural_effusion"


def check_row(row, frontal_images, lateral_images):
    try:
        batch_id = row["batch_id"]
        labels = [label.strip() for label in row[LABELS_COLUMN_NAME].split(",")]
        image_paths = ast.literal_eval(row["image_paths"])
    except Exception as e:
        return None

    if batch_id != TARGET_BATCH_ID:
        return None

    if labels != [TARGET_LABEL_NAME]:
        return None

    if isinstance(image_paths, dict):
        image_paths_dict = image_paths
        image_paths = []
        for value in image_paths_dict.values():
            image_paths.extend(value["paths"])

    if len(image_paths) != 2:
        return None

    deid_dir = "deid" if batch_id in ("20DEC2024", "09JAN2025") else ""
    image_paths = [os.path.join(GCS_INPUT_IMAGES_DIR, batch_id, deid_dir, image_path) for image_path in image_paths]

    if not any(image_path in frontal_images for image_path in image_paths) or not any(image_path in lateral_images for image_path in image_paths):
        return None

    sorted_image_paths = [image_path for image_path in image_paths if image_path in frontal_images] + [image_path for image_path in image_paths if image_path in lateral_images]

    return {"image_path": sorted_image_paths, "labels": labels}


def main():
    print("Downloading frontal projections file")
    gcs_data = gcs_utils.split_gcs_uri(GCS_FRONTAL_PROJECTIONS_FILE)
    content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

    print("Generating a list of frontal images")
    df = pd.read_csv(StringIO(content), header=None, sep=';')
    frontal_images = set(df[0])

    print("Downloading lateral projections file")
    gcs_data = gcs_utils.split_gcs_uri(GCS_LATERAL_PROJECTIONS_FILE)
    content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

    print("Generating a list of lateral images")
    df = pd.read_csv(StringIO(content), header=None, sep=';')
    lateral_images = set(df[0])

    print("Downloading input file")
    gcs_data = gcs_utils.split_gcs_uri(GCS_INPUT_FILE)
    content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

    print("Loading input file")
    df = pd.read_csv(StringIO(content), low_memory=False)
    df = df[[LABELS_COLUMN_NAME, "image_paths", "batch_id"]]

    print("Generating a list of target images")
    target_images = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        row_data = check_row(row, frontal_images, lateral_images)
        if row_data is None:
            continue

        target_images.append(row_data)

    print(f"Found {len(target_images)} target images")

    os.makedirs(DESTINATION_DIR, exist_ok=True)

    print(f"Copying {TARGET_NUM_IMAGES} frontal images to the destination folder")
    for target_image in target_images[:TARGET_NUM_IMAGES]:
        image_path = target_image["image_path"][0]
        image_path = image_path.replace(GCS_INPUT_IMAGES_DIR, TARGET_IMAGES_DIR)
        image = dicom_utils.get_dicom_image(image_path, custom_windowing_parameters={"window_center": 0, "window_width": 0})
        image = image_utils.numpy_array_to_pil_image(image, convert_to_uint8=True, convert_to_rgb=True)
        new_image_path = os.path.join(DESTINATION_DIR, os.path.basename(image_path).replace(".dcm", ".png"))
        image.save(new_image_path)


if __name__ == "__main__":
    main()
