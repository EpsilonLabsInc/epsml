import ast
import os
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from epsutils.dicom import dicom_utils
from epsutils.gcs import gcs_utils
from epsutils.training.confusion_matrix_calculator import ConfusionMatrixCalculator
from epsclassifiers.intern_vit_classifier import InternVitClassifier

INITIAL_CHECKPOINT_DIR = "/workspace/models/old/internvl2.5_26b_finetune_lora_20241229_184000_1e-5_2.5_gradient_full_rm_sole_no_findings_rm_bad_dcm_no_label/checkpoint-58670"
TRAINING_CHECKPOINT = "/workspace/models/intern_vit_classifier-training-on-gradient_cr_pneumothorax/intern_vit_pneumothorax_checkpoint_epoch_4_20250129_203157.pt"
GCS_VALIDATION_FILE = "gs://gradient-crs/archive/training/gradient-crs-09JAN2025-per-study-chest-images-with-pneumothorax-label-validation-all.jsonl"
SOURCE_IMAGE_PATH = "gs://epsilon-data-us-central1/GRADIENT-DATABASE/CR/09JAN2025/deid"
TARGET_IMAGE_PATH = "/workspace/CR/09JAN2025"
VALIDATE_POSITIVE_ONLY = False
MAX_VALIDATION_COUNT = None


def main():
    print("Creating model")
    classifier = InternVitClassifier(num_classes=1, intern_vl_checkpoint_dir=INITIAL_CHECKPOINT_DIR, intern_vit_output_dim=3200)  # 3200 for InternVL 26B model, 1024 for InternVL 8B model.
    training_checkpoint = torch.load(TRAINING_CHECKPOINT)
    classifier.load_state_dict(training_checkpoint["model_state_dict"])
    classifier = classifier.to("cuda")
    classifier.eval()
    image_processor = classifier.get_image_processor()

    print(f"Downloading validation file {GCS_VALIDATION_FILE}")
    gcs_data = gcs_utils.split_gcs_uri(GCS_VALIDATION_FILE)
    content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

    y_true = []
    y_pred = []
    lines = content.splitlines()

    with torch.no_grad():
        for line in tqdm(lines, total=len(lines), desc="Processing"):
            line = ast.literal_eval(line)

            image_paths = line["image_path"]
            labels = line["labels"]

            if VALIDATE_POSITIVE_ONLY and not labels:
                continue

            if MAX_VALIDATION_COUNT and len(y_pred) >= MAX_VALIDATION_COUNT:
                break

            res = []

            for image_path in image_paths:
                if SOURCE_IMAGE_PATH and TARGET_IMAGE_PATH and image_path.startswith(SOURCE_IMAGE_PATH):
                    image_path = os.path.join(TARGET_IMAGE_PATH, os.path.relpath(image_path, SOURCE_IMAGE_PATH))

                if gcs_utils.is_gcs_uri(image_path):
                    gcs_data = gcs_utils.split_gcs_uri(image_path)
                    data = BytesIO(gcs_utils.download_file_as_bytes(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"]))
                else:
                    data = image_path

                image = dicom_utils.get_dicom_image(data, custom_windowing_parameters={"window_center": 0, "window_width": 0})
                image = image.astype(np.float32)
                eps = 1e-10
                image = (image - image.min()) / (image.max() - image.min() + eps) * 255
                image = image.astype(np.uint8)
                image = Image.fromarray(image)
                image = image.convert("RGB")

                pixel_values = image_processor(images=[image, image], return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(torch.bfloat16).cuda()

                output = classifier(pixel_values)[0]
                probabilities = torch.sigmoid(output)
                assert probabilities.shape == (1,)
                res.append(probabilities[0] >= 0.5)

                del pixel_values
                del output
                del probabilities
                torch.cuda.empty_cache()

            y_true.append(1 if len(labels) > 0 else 0)
            y_pred.append(1 if any(res) else 0)

            if len(y_pred) % 20 == 0:
                calc = ConfusionMatrixCalculator()
                cm = calc.compute_confusion_matrix(y_true=y_true, y_pred=y_pred)
                print("")
                print("Confusion matrix:")
                print(cm)

    calc = ConfusionMatrixCalculator()
    cm = calc.compute_confusion_matrix(y_true=y_true, y_pred=y_pred)
    print("")
    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()
