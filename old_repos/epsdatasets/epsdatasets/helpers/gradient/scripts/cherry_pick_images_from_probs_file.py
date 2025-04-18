import ast
import json
import os

from tqdm import tqdm

from epsutils.dicom import dicom_utils
from epsutils.gcs import gcs_utils
from epsutils.image import image_utils
from epsutils.sys import sys_utils

INPUT_FILE = "/home/andrej/tmp/probs/pleural_effusion_misclassified_epoch_4_20250314_035046_utc.jsonl"
TARGET_VALUE_TO_MATCH = [0.0]
OUTPUT_VALUE_TO_MATCH = [1]
PATH_SUBSTITUTIONS = {
    "/workspace/CR/22JUL2024/": "/mnt/efs/all-cxr/gradient/22JUL2024/",
    "/workspace/CR/20DEC2024/": "/mnt/efs/all-cxr/gradient/20DEC2024/deid/",
    "/workspace/CR/09JAN2025/": "/mnt/efs/all-cxr/gradient/09JAN2025/deid/",
}
NUM_IMAGES_TO_CHERRY_PICK = 40
DESTINATION_DIR = "/home/andrej/tmp/pleural_effusion"
NOTES = "False positives only"


def main():
    # Load input file.
    print("Loading input file")
    content = open(INPUT_FILE, "r", encoding="utf-8")

    # Cherry pick the images.
    print("Cherry picking the images")
    selected_rows = []
    for row in content:
        row = ast.literal_eval(row)
        target = row["target"]
        output = row["output"]

        if target == TARGET_VALUE_TO_MATCH and output == OUTPUT_VALUE_TO_MATCH:
            selected_rows.append(row)

        if len(selected_rows) >= NUM_IMAGES_TO_CHERRY_PICK:
            break

    # Create destination dir.
    os.makedirs(DESTINATION_DIR, exist_ok=True)

    # Copy selected images to destination dir.
    print(f"Copying {len(selected_rows)} frontal images to the destination folder")
    for row in tqdm(selected_rows, total=len(selected_rows)):
        image_path = row["file_name"]
        image_path = image_path[0] if isinstance(image_path, list) else image_path
        image_path = sys_utils.apply_path_substitutions(image_path, PATH_SUBSTITUTIONS)

        image = dicom_utils.get_dicom_image(image_path, custom_windowing_parameters={"window_center": 0, "window_width": 0})
        image = image_utils.numpy_array_to_pil_image(image, convert_to_uint8=True, convert_to_rgb=True)
        new_image_path = os.path.join(DESTINATION_DIR, os.path.basename(image_path).replace(".dcm", ".jpg"))
        image.save(new_image_path)

    # Save info.
    info_file = os.path.join(DESTINATION_DIR, "info.jsonl")
    with open(info_file, "w", encoding="utf-8") as file:
        for row in selected_rows:
            file.write(json.dumps(row) + "\n")

    # Save notes.
    notes_file = os.path.join(DESTINATION_DIR, "notes.txt")
    with open(notes_file, "w", encoding="utf-8") as file:
        file.write(NOTES)


if __name__ == "__main__":
    main()
