# General config.
REPORTS_FILE = "/home/andrej/data/gradient_ct_batch_1_chest_only.csv"
CHECK_BODY_PARTS_OUTPUT_FILE = "/home/andrej/data/bodypart/body-part-ct-16AGO2024-log.txt"
CONVERT_TO_EPSILON_BODY_PARTS_OUTPUT_FILE = "/home/andrej/data/bodypart/body-part-ct-16AGO2024.csv"
VALIDATION_OUTPUT_FILE = "/home/andrej/data/bodypart/csv-body-part-validation-ct-16AGO2024.csv"
SKIP_ALREADY_PROCESSED_IMAGES = True
USE_CPU_FOR_FAIL_SAFE = False
REJECT_NON_AXIAL = True

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
Reports file: {REPORTS_FILE}
Check body parts output file: {CHECK_BODY_PARTS_OUTPUT_FILE}
Convert to Epsilon body parts output file: {CONVERT_TO_EPSILON_BODY_PARTS_OUTPUT_FILE}
Validation output file: {VALIDATION_OUTPUT_FILE}
Skip already processed images: {SKIP_ALREADY_PROCESSED_IMAGES}
Use CPU for fail-safe: {USE_CPU_FOR_FAIL_SAFE}
Reject non-axial: {REJECT_NON_AXIAL}
GCS bucket name: {GCS_BUCKET_NAME}
GCS images dir: {GCS_IMAGES_DIR}
Num preprocessing workers: {NUM_PREPROCESSING_WORKERS}
Max queue size: {MAX_QUEUE_SIZE}
Empty queue wait timeout (sec): {EMPTY_QUEUE_WAIT_TIMEOUT_SEC}
------------------------------------------------------""")
