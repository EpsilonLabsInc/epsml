import logging
import multiprocessing
import os
import queue
import threading
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from epsutils.gcs import gcs_utils
from epsutils.segmentation.monai.monai_segmentator import MonaiSegmentator

import config

manager = multiprocessing.Manager()
gpu_queue = manager.Queue()
segmentator = MonaiSegmentator()


def start(nifti_files):
    inference_thread = threading.Thread(target=inference_task)
    inference_thread.start()

    with ProcessPoolExecutor(max_workers=config.NUM_PREPROCESSING_WORKERS) as executor:
        results = list(tqdm(executor.map(preprocessing_task, nifti_files), total=len(nifti_files), desc="Preprocessing"))

    inference_thread.join()

def preprocessing_task(nifti_file):
    try:
        gcs_nifti_file = os.path.join(config.GCS_IMAGES_DIR, nifti_file)
        gcs_utils.download_file(gcs_bucket_name=config.GCS_BUCKET_NAME,
                                gcs_file_name=gcs_nifti_file,
                                local_file_name=nifti_file,
                                num_retries=None,  # Retry indefinitely.
                                show_warning_on_retry=True)

        if not os.path.exists(nifti_file):
            err_msg = f"ERROR: NIfTI file not properly downloaded ({gcs_nifti_file})"
            print(err_msg)
            logging.error(err_msg)
            return

        data = segmentator.preprocessing(file_or_dir=nifti_file)
        gpu_queue.put(data)
    except Exception as e:
        err_msg = f"ERROR: {str(e)} ({gcs_nifti_file})"
        print(err_msg)
        logging.error(err_msg)
    finally:
        if os.path.exists(nifti_file):
            os.remove(nifti_file)

def inference_task():
    while True:
        gcs_nifti_file = None

        try:
            data = gpu_queue.get(block=True, timeout=config.EMPTY_QUEUE_WAIT_TIMEOUT_SEC)
            gcs_nifti_file = os.path.join(config.GCS_IMAGES_DIR, data["image"].meta["filename_or_obj"])
            res = segmentator.segmentation(data=data)
            logging.info(f"{res['info']},{gcs_nifti_file}")
        except queue.Empty:
            # Stop task when GPU queue empty.
            break
        except Exception as e:
            err_msg = f"ERROR: {str(e)} ({gcs_nifti_file})"
            print(err_msg)
            logging.error(err_msg)
