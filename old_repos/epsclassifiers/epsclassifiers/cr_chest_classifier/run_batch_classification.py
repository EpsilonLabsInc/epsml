import ast
import logging
import multiprocessing
import os
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import torch
from tqdm import tqdm

from epsutils.dicom import dicom_utils
from epsutils.logging import logging_utils

from cr_chest_classifier import CrChestClassifier, Label

INPUT_FILE_NAME = "/mnt/efs/all-cxr/combined/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_all.csv"
IMAGE_PATH_COLUMN_NAME = "image_paths"
BASE_PATH_COLUMN_NAME = "base_path"
OUTPUT_FILE = "gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_all_chest_non_chest.csv"
MAX_BATCH_SIZE = 192
EMPTY_QUEUE_WAIT_TIMEOUT_SEC = 300
BASE_PATH_SUBSTITUTIONS = {
    "gradient/22JUL2024": "/mnt/efs/all-cxr/gradient/22JUL2024",
    "gradient/20DEC2024": "/mnt/efs/all-cxr/gradient/20DEC2024/deid",
    "gradient/09JAN2025": "/mnt/efs/all-cxr/gradient/09JAN2025/deid",
    "gradient/16AUG2024": None,
    "gradient/13JAN2025": None,
    "segmed/batch1": "/mnt/efs/all-cxr/segmed/batch1",
    "segmed/batch2": "/mnt/efs/all-cxr/segmed/batch2",
    "segmed/batch3": "/mnt/efs/all-cxr/segmed/batch3",
    "segmed/batch4": "/mnt/efs/all-cxr/segmed/batch4",
    "simonmed": "/mnt/efs/all-cxr/simonmed/images/422ca224-a9f2-4c64-bf7c-bb122ae2a7bb"
}


manager = multiprocessing.Manager()
dicom_queue = manager.Queue()
classifier = CrChestClassifier()
classifier.load_state_dict(torch.load("models/cr_chest_classifier_trained_on_600k_gradient_samples.pt"))


def load_dicom_file(df_row):
    try:
        # Get base path.
        base_path = df_row[BASE_PATH_COLUMN_NAME]
        if base_path not in BASE_PATH_SUBSTITUTIONS:
            raise ValueError(f"Base path '{base_path}' not in base path substitutions")
        elif BASE_PATH_SUBSTITUTIONS[base_path] is None:
            return

        # Get image paths.
        try:
            image_paths = ast.literal_eval(df_row[IMAGE_PATH_COLUMN_NAME])
        except:
            image_paths = df_row[IMAGE_PATH_COLUMN_NAME]
        image_paths = image_paths if isinstance(image_paths, list) else [image_paths]

        # Load images.
        image_paths = [os.path.join(BASE_PATH_SUBSTITUTIONS[base_path], image_path) for image_path in image_paths]
        images = [dicom_utils.get_dicom_image_fail_safe(image_path, custom_windowing_parameters={"window_center": 0, "window_width": 0}) for image_path in image_paths]

        # Wait if queue full.
        while dicom_queue.qsize() >= MAX_BATCH_SIZE * 20:
            time.sleep(0.5)

        # Add to queue.
        for image, image_path in zip(images, image_paths):
            dicom_queue.put({"dicom_file": image_path, "image": image})
    except Exception as e:
        print(f"Error loading DICOM file: {str(e)} ({df_row[IMAGE_PATH_COLUMN_NAME]})")


def classification_task(progress_bar):
    print("Classification task started")

    while True:
        try:
            # Get MAX_BATCH_SIZE DICOM files from the queue.
            dicom_files = []
            images = []
            for _ in range(MAX_BATCH_SIZE):
                item = dicom_queue.get(block=True, timeout=EMPTY_QUEUE_WAIT_TIMEOUT_SEC)
                dicom_files.append(item["dicom_file"])
                images.append(item["image"])

            # Predict.
            labels = classifier.predict(images=images, device="cuda")

            if len(dicom_files) != len(labels):
                raise ValueError(f"Number of DICOM files ({len(dicom_files)}) differs from number of labels ({len(labels)})")

            # Write results.
            for dicom_file, label in zip(dicom_files, labels):
                slabel = "CHEST" if label == Label.CHEST else "NON-CHEST"
                logging.info(f"{dicom_file},{slabel}")

        except queue.Empty:
            # DICOM queue is empty. Classify the remaining DICOM files in the queue and exit the classification loop.
            if len(dicom_files) > 0:
                labels = classifier.predict(images=images, device="cuda")
                for dicom_file, label in zip(dicom_files, labels):
                    slabel = "CHEST" if label == Label.CHEST else "NON-CHEST"
                    logging.info(f"{dicom_file},{slabel}")

            break

        except Exception as e:
            print(f"Error during classification: {str(e)}")

        finally:
            progress_bar.update(len(dicom_files))

    print("Classification task finished")


def main():
    logging_utils.configure_logger(logger_file_name=OUTPUT_FILE, show_logging_level=False)

    print(f"Loading {INPUT_FILE_NAME}")
    df = pd.read_csv(INPUT_FILE_NAME, low_memory=False)

    print("Estimating number of images that will be classified")
    num_images = 0
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        base_path = row[BASE_PATH_COLUMN_NAME]
        if base_path not in BASE_PATH_SUBSTITUTIONS or BASE_PATH_SUBSTITUTIONS[base_path] is None:
            continue

        try:
            image_paths = ast.literal_eval(row[IMAGE_PATH_COLUMN_NAME])
        except:
            image_paths = row[IMAGE_PATH_COLUMN_NAME]
        image_paths = image_paths if isinstance(image_paths, list) else [image_paths]
        num_images += len(image_paths)

    progress_bar = tqdm(total=num_images, leave=False, desc="Processing")
    classification_thread = threading.Thread(target=classification_task, args=(progress_bar,))
    classification_thread.start()

    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(load_dicom_file, [row for _, row in df.iterrows()]))

    print("All DICOM files loaded")

    classification_thread.join()
    progress_bar.close()


if __name__ == "__main__":
    main()
