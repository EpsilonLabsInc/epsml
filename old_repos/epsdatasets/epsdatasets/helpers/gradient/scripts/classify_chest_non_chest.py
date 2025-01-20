import inspect
import logging
import multiprocessing
import os
import queue
import threading
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO

import pydicom
import torch
from tqdm import tqdm

from epsclassifiers.cr_chest_classifier import CrChestClassifier, Label
from epsutils.dicom import dicom_utils
from epsutils.gcs import gcs_utils
from epsutils.logging import logging_utils

EPSILON_GCS_BUCKET_NAME = "gradient-crs"
EPSILON_GCS_IMAGES_DIR = "20DEC2024"
GRADIENT_GCS_BUCKET_NAME = "epsilon-data-us-central1"
GRADIENT_GCS_IMAGES_DIR = "GRADIENT-DATABASE/CR/20DEC2024/deid"
CHECK_IF_BODY_PART_EXAMINED_IS_CHEST = True
BODY_PART_EXAMINED_DICOM_TAG = "(0018,0015) Body Part Examined:"
OUTPUT_FILE = "output/gradient-crs-20DEC2024-chest_non_chest.csv"
MAX_BATCH_SIZE = 16
EMPTY_QUEUE_WAIT_TIMEOUT_SEC = 60

MODEL_PATH = os.path.join(os.path.dirname(inspect.getmodule(CrChestClassifier).__file__), "models/cr_chest_classifier_trained_on_600k_gradient_samples.pt")

manager = multiprocessing.Manager()
dicom_queue = manager.Queue()
classifier = CrChestClassifier()
classifier.load_state_dict(torch.load(MODEL_PATH))


def is_body_part_examined_chest(txt_file):
    try:
        txt_file = os.path.join(EPSILON_GCS_IMAGES_DIR, txt_file)
        content = gcs_utils.download_file_as_string(gcs_bucket_name=EPSILON_GCS_BUCKET_NAME, gcs_file_name=txt_file)

        rows = content.splitlines()
        for row in rows:
            if row.startswith(BODY_PART_EXAMINED_DICOM_TAG):
                body_part = row[len(BODY_PART_EXAMINED_DICOM_TAG):].strip()
                return body_part in ["Chest", "Chest/Abdomen"]

        raise ValueError("BodyPartExamined DICOM tag not found")

    except Exception as e:
        print(f"Error downloading/parsing TXT file: {str(e)} ({txt_file})")
        raise


def download_dicom_file(txt_file):
    if CHECK_IF_BODY_PART_EXAMINED_IS_CHEST and not is_body_part_examined_chest(txt_file):
        return

    try:
        dicom_file = txt_file.replace("_", "/").replace(".txt", ".dcm")
        dicom_file = os.path.join(GRADIENT_GCS_IMAGES_DIR, dicom_file)
        content = gcs_utils.download_file_as_bytes(gcs_bucket_name=GRADIENT_GCS_BUCKET_NAME, gcs_file_name=dicom_file)
        dataset = pydicom.dcmread(BytesIO(content))
        image = dicom_utils.get_dicom_image_from_dataset(dataset, custom_windowing_parameters={"window_center": 0, "window_width": 0})
        dicom_queue.put({"dicom_file": dicom_file, "image": image})
    except Exception as e:
        print(f"Error downloading DICOM file: {str(e)} ({dicom_file})")


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
                logging.info(f"{dicom_file};{slabel}")

        except queue.Empty:
            # DICOM queue is empty. Classify the remaining DICOM files in the queue and exit the classification loop.
            if len(dicom_files) > 0:
                labels = classifier.predict(images=images, device="cuda")
                for dicom_file, label in zip(dicom_files, labels):
                    slabel = "CHEST" if label == Label.CHEST else "NON-CHEST"
                    logging.info(f"{dicom_file};{slabel}")

            break

        except Exception as e:
            print(f"Error during classification: {str(e)}")

        finally:
            progress_bar.update(len(dicom_files))

    print("Classification task finished")


def main():
    logging_utils.configure_logger(logger_file_name=OUTPUT_FILE, show_logging_level=False)

    # Get a list of all files in the GCS bucket.
    print(f"Getting a list of all files in the {EPSILON_GCS_BUCKET_NAME} GCS bucket")
    all_files_in_bucket = gcs_utils.list_files(gcs_bucket_name=EPSILON_GCS_BUCKET_NAME, gcs_dir=EPSILON_GCS_IMAGES_DIR)

    # Select TXT files.
    txt_files_in_bucket = [os.path.basename(file) for file in all_files_in_bucket if file.endswith(".txt")]
    txt_files_in_bucket = set(txt_files_in_bucket)  # To avoid duplicate files.
    print(f"Total TXT files found: {len(txt_files_in_bucket)}")

    # Start classification task.
    progress_bar = tqdm(total=len(txt_files_in_bucket), leave=False, desc="Processing")
    classification_thread = threading.Thread(target=classification_task, args=(progress_bar,))
    classification_thread.start()

    # Start DICOM downloaders.
    with ProcessPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(download_dicom_file, [txt_file for txt_file in txt_files_in_bucket]))

    print("All DICOM files downloaded")

    classification_thread.join()
    progress_bar.close()


if __name__ == "__main__":
    main()
