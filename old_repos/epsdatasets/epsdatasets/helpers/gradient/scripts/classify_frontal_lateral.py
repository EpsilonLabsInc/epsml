import inspect
import logging
import multiprocessing
import os
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor

import pydicom
import torch
from tqdm import tqdm

from epsclassifiers.cr_projection_classifier import CrProjectionClassifier, LABEL_TO_STRING
from epsutils.gcs import gcs_utils
from epsutils.logging import logging_utils

EPSILON_GCS_IMAGES_DIR = "gs://gradient-crs/22JUL2024"
GRADIENT_GCS_IMAGES_DIR = "/workspace/CR/22JUL2024"
GRADIENT_DIR_PREFIX_TO_REMOVE = "/workspace/CR"
OUTPUT_FILE = "gradient-crs-22JUL2024-frontal-lateral.csv"
MAX_BATCH_SIZE = 64
EMPTY_QUEUE_WAIT_TIMEOUT_SEC = 60

MODEL_PATH = os.path.join(os.path.dirname(inspect.getmodule(CrProjectionClassifier).__file__), "models/cr_projection_classifier_trained_on_500k_gradient_samples.pt")

manager = multiprocessing.Manager()
dicom_queue = manager.Queue()
classifier = CrProjectionClassifier()
classifier.load_state_dict(torch.load(MODEL_PATH))


def download_dicom_file(txt_file):
    try:
        dicom_file = txt_file.replace("_", "/").replace(".txt", ".dcm")
        dicom_file = os.path.join(GRADIENT_GCS_IMAGES_DIR, dicom_file)
        dataset = pydicom.dcmread(dicom_file)
        image = classifier.preprocess(dataset)

        while dicom_queue.qsize() >= MAX_BATCH_SIZE * 20:
            time.sleep(0.5)

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
                logging.info(f"{os.path.relpath(dicom_file, GRADIENT_DIR_PREFIX_TO_REMOVE)};{LABEL_TO_STRING[label]}")

        except queue.Empty:
            # DICOM queue is empty. Classify the remaining DICOM files in the queue and exit the classification loop.
            if len(dicom_files) > 0:
                labels = classifier.predict(images=images, device="cuda")
                for dicom_file, label in zip(dicom_files, labels):
                    logging.info(f"{os.path.relpath(dicom_file, GRADIENT_DIR_PREFIX_TO_REMOVE)};{LABEL_TO_STRING[label]}")

            break

        except Exception as e:
            print(f"Error during classification: {str(e)}")

        finally:
            progress_bar.update(len(dicom_files))

    print("Classification task finished")


def main():
    logging_utils.configure_logger(logger_file_name=OUTPUT_FILE, show_logging_level=False)

    print(f"Getting a list of all TXT files in {EPSILON_GCS_IMAGES_DIR}")
    gcs_data = gcs_utils.split_gcs_uri(EPSILON_GCS_IMAGES_DIR)
    files_in_bucket = gcs_utils.list_files(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_dir=gcs_data["gcs_path"])
    txt_files_in_bucket = [os.path.basename(file) for file in files_in_bucket if file.endswith(".txt")]
    txt_files_in_bucket = set(txt_files_in_bucket)  # To avoid duplicate files.
    print(f"Total TXT files found: {len(txt_files_in_bucket)}")

    progress_bar = tqdm(total=len(txt_files_in_bucket), leave=False, desc="Processing")
    classification_thread = threading.Thread(target=classification_task, args=(progress_bar,))
    classification_thread.start()

    with ProcessPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(download_dicom_file, [txt_file for txt_file in txt_files_in_bucket]))

    print("All DICOM files downloaded")

    classification_thread.join()
    progress_bar.close()


if __name__ == "__main__":
    main()
