# General config.
DISPLAY_NAME = "body-part-ct-16AGO2024"
OUTPUT_DIR = "/home/andrej/data/bodypart"
SKIP_ALREADY_PROCESSED_IMAGES = True

# GCS config.
GCS_BUCKET_NAME = "gradient-cts-fixed-nifti"
GCS_IMAGES_DIR = "16AGO2024"

# Processing.
NUM_PREPROCESSING_WORKERS = 12
MAX_QUEUE_SIZE = 50
EMPTY_QUEUE_WAIT_TIMEOUT_SEC = 60


def dump_config():
    print(f"""Using the following configuration settings:
------------------------------------------------------
Display name: {DISPLAY_NAME}
Output dir: {OUTPUT_DIR}
GCS bucket name: {GCS_BUCKET_NAME}
GCS images dir: {GCS_IMAGES_DIR}
Num preprocessing workers: {NUM_PREPROCESSING_WORKERS}
Max queue size: {MAX_QUEUE_SIZE}
Empty queue wait timeout (sec): {EMPTY_QUEUE_WAIT_TIMEOUT_SEC}
------------------------------------------------------""")
