import ast
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from io import BytesIO

from tqdm import tqdm

from epsutils.dicom import dicom_utils
from epsutils.gcs import gcs_utils
from epsutils.logging import logging_utils


IMAGES_FILE = "gs://epsilonlabs-filestore/cleaned_CRs/gradient_rm_bad_dcm_1211_nolabel.jsonl"
DESTINATION_IMAGES_DIR = "/mnt/gradient/gradient-cxr/22JUL2024"
OUTPUT_FILE = "./output/gradient_rm_bad_dcm_1211_nolabel-corrupt-images.csv"


class FileType(Enum):
    JSONL = 1
    UNSUPPORTED = 2


def get_file_type(file_name):
    if file_name.endswith(".jsonl"):
        return FileType.JSONL
    else:
        return FileType.UNSUPPORTED


def load_file(file_name):
    if gcs_utils.is_gcs_uri(file_name):
        gcs_data = gcs_utils.split_gcs_uri(file_name)
        content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])
        return content
    else:
        raise ValueError(f"Load option not implemented")


def parse_content(file_type, content):
    if file_type == FileType.JSONL:
        return parse_jsonl(content)
    else:
        raise ValueError("Parse option not implemented")


def parse_jsonl(content):
    images = []
    rows = content.splitlines()

    for row in rows:
        row = ast.literal_eval(row)
        image_list = row["image"]

        for image in image_list:
            if image.endswith(".txt"):
                image = image.replace("_", "/").replace(".txt", ".dcm")
            image_path = os.path.join(DESTINATION_IMAGES_DIR, image) if DESTINATION_IMAGES_DIR else image
            images.append(image_path)

    return images


def check_image(image_path):
    if gcs_utils.is_gcs_uri(image_path):
        gcs_data = gcs_utils.split_gcs_uri(image_path)
        content = gcs_utils.download_file_as_bytes(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])
        content = BytesIO(content)
    else:
        content = image_path

    try:
        image = dicom_utils.get_dicom_image(content, custom_windowing_parameters={"window_center": 0, "window_width": 0})
    except Exception as e:
        print(f"Corrupt image {image_path}: {str(e)}")
        logging.warning(image_path)


def main():
    output_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logging_utils.configure_logger(logger_file_name=OUTPUT_FILE, show_logging_level=False)

    file_type = get_file_type(IMAGES_FILE)
    if file_type == FileType.UNSUPPORTED:
        raise ValueError("File type is unsupported")

    print(f"Loading images file")
    content = load_file(IMAGES_FILE)

    print("Parsing content")
    images = parse_content(file_type, content)

    with ProcessPoolExecutor() as executor:  # Use default number of workers.
        results = list(tqdm(executor.map(check_image, [image for image in images]), total=len(images), desc="Processing"))


if __name__ == "__main__":
    main()
