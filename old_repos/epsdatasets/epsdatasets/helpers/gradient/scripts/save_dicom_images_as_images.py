import ast
import os
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from epsutils.dicom import dicom_utils
from epsutils.gcs import gcs_utils
from epsutils.image import image_utils

GCS_REPORTS_FILE = "gs://gradient-crs/archive/training/all-labels/gradient-crs-22JUL2024-chest-images-with-labels-training.jsonl"
DIR_PREFIX_TO_REMOVE = "GRADIENT-DATABASE/CR/"
REMOVE_DEID_FROM_PATH = False
DIR_PREFIX_TO_ADD = "/mnt/efs/all-cxr/gradient/"
DIR_PREFIX_TO_SAVE = "/mnt/efs/all-cxr/gradient-png/"
TARGET_IMAGE_SIZE = (448, 448)
IMAGE_EXTENSION_TO_SAVE = "png"


def save_image(image_path):
    image = dicom_utils.get_dicom_image(image_path, custom_windowing_parameters={"window_center": 0, "window_width": 0})
    image = image_utils.numpy_array_to_pil_image(image, convert_to_uint8=True, convert_to_rgb=True)
    image = image.resize(TARGET_IMAGE_SIZE)

    new_image_path = os.path.relpath(image_path, DIR_PREFIX_TO_ADD)
    new_image_path = os.path.join(DIR_PREFIX_TO_SAVE, new_image_path)
    new_image_path = new_image_path.replace("dcm", IMAGE_EXTENSION_TO_SAVE)

    os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
    image.save(new_image_path)


def main():
    print(f"Downloading reports file {GCS_REPORTS_FILE}")
    gcs_data = gcs_utils.split_gcs_uri(GCS_REPORTS_FILE)
    content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

    print("Reading reports file")
    image_paths = []
    rows = content.splitlines()
    for index, row in tqdm(enumerate(rows), total=len(rows)):
        row = ast.literal_eval(row)
        image_path = row["image_path"]
        image_path = os.path.relpath(image_path, DIR_PREFIX_TO_REMOVE) if DIR_PREFIX_TO_REMOVE else image_path
        if REMOVE_DEID_FROM_PATH:
            image_path = image_path.replace("/deid/", "/")
        image_path = os.path.join(DIR_PREFIX_TO_ADD, image_path)
        image_paths.append(image_path)

    with ProcessPoolExecutor() as executor:  # Use default number of workers.
        results = list(tqdm(executor.map(save_image, [image_path for image_path in image_paths]), total=len(image_paths), desc="Saving images"))


if __name__ == "__main__":
    main()
