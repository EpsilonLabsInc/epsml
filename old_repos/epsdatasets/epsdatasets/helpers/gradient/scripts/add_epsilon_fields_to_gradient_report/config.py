import pydicom


SOURCE_GCS_BUCKET_NAME = "epsilon-data-us-central1"
SOURCE_GCS_REPORTS_FILE = "GRADIENT-DATABASE/REPORTS/CT/ct-18set2024-batch-2.csv"

NIFTI_FILES_GCS_BUCKET_NAME = "gradient-cts-nifti"
NIFTI_FILES_GCS_IMAGES_DIR = "18SET2024"
LOCAL_IMAGES_DIR = None

DESTINATION_GCS_BUCKET_NAME = "gradient-cts-nifti"
DESTINATION_GCS_REPORTS_FILE = "ct-18set2024-batch-2_studyAccepted_primaryVolumes.csv"
