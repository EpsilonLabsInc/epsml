import ast
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from io import StringIO

import pandas as pd
from PIL import Image
from tqdm import tqdm

from epsutils.dicom import dicom_utils
from epsutils.gcs import gcs_utils
from epsutils.image import image_utils
from epsutils.logging import logging_utils

GCS_REPORTS_FILE = "gs://report_csvs/cleaned/CR/GRADIENT_CR_22JUL2024.csv"
GCS_CHEST_IMAGES_FILE = "gs://gradient-crs/archive/training/chest/chest_files_gradient_all_3_batches.csv"
SOURCE_IMAGES_DIR = "GRADIENT-DATABASE/CR/22JUL2024"
DIR_PREFIX_TO_ADD = "/mnt/efs/all-cxr/gradient/22JUL2024"
DIR_PREFIX_TO_SAVE = "/mnt/efs/all-cxr/gradient-png/22JUL2024"
TARGET_IMAGE_SIZE = (448, 448)
IMAGE_EXTENSION_TO_SAVE = "png"
LOG_FILE = "save_dicom_images_as_images.log"


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


def save_image(image_path):
    try:
        new_image_path = os.path.relpath(image_path, DIR_PREFIX_TO_ADD)
        new_image_path = os.path.join(DIR_PREFIX_TO_SAVE, new_image_path)
        new_image_path = new_image_path.replace("dcm", IMAGE_EXTENSION_TO_SAVE)

        image = dicom_utils.get_dicom_image(image_path, custom_windowing_parameters={"window_center": 0, "window_width": 0})

        res, err = image_utils.validate_image_histogram(image=Image.fromarray(image), config=image_utils.VALIDATE_IMAGE_HISTOGRAM_CONFIGURATIONS["CHEST_CR_SCAN"])
        if not res:
            logging.warning(f"Histogram validation failed: {err}")
            return

        image = image_utils.numpy_array_to_pil_image(image, convert_to_uint8=True, convert_to_rgb=True)
        image = image.resize(TARGET_IMAGE_SIZE)

        os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
        image.save(new_image_path)

        logging.info(f"Image successfully saved: {new_image_path}")
    except Exception as e:
        print(f"Error saving {image_path}: {str(e)}")


def main():
    logging_utils.configure_logger(logger_file_name=LOG_FILE, show_logging_level=True)

    print("Downloading chest images file")
    gcs_data = gcs_utils.split_gcs_uri(GCS_CHEST_IMAGES_FILE)
    content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

    print("Reading a list of chest images")
    df = pd.read_csv(StringIO(content), header=None, sep=';')
    chest_images = set(df[0])

    print(f"Downloading reports file {GCS_REPORTS_FILE}")
    gcs_data = gcs_utils.split_gcs_uri(GCS_REPORTS_FILE)
    content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

    print("Reading reports file")
    df = pd.read_csv(StringIO(content), low_memory=False)

    all_image_paths = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            image_paths = ast.literal_eval(row["image_paths"])
            age = row["age"]
        except Exception as e:
            continue

        parsed_age = get_age(age)

        if parsed_age is None:
            continue

        if parsed_age < 6:
            logging.warning(f"Age < 6")
            continue

        if isinstance(image_paths, dict):
            image_paths_dict = image_paths
            image_paths = []
            for value in image_paths_dict.values():
                image_paths.extend(value["paths"])

        for image_path in image_paths:
            full_image_path = os.path.join(SOURCE_IMAGES_DIR, image_path)
            if full_image_path not in chest_images:
                continue

            image_path = os.path.join(DIR_PREFIX_TO_ADD, image_path)
            all_image_paths.append(image_path)

    with ProcessPoolExecutor() as executor:  # Use default number of workers.
        results = list(tqdm(executor.map(save_image, [image_path for image_path in all_image_paths]), total=len(all_image_paths), desc="Saving images"))


if __name__ == "__main__":
    main()
