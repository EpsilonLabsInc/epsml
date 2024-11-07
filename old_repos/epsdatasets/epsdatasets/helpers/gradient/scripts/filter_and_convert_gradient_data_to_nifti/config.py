import pydicom


# General config.
DISPLAY_NAME = "gradient-mr-01OCT2024"
OUTPUT_DIR = "/home/andrej/data/gradient/output"
SKIP_EXISTING_FILES = False

# GCS config.
SOURCE_GCS_BUCKET_NAME = "epsilon-data-us-central1"
SOURCE_GCS_REPORTS_FILE = "GRADIENT-DATABASE/REPORTS/MR/epsilon-mri-final-01oct2024_v2.csv"
SOURCE_GCS_IMAGES_DIR = "GRADIENT-DATABASE/MR"
DESTINATION_GCS_BUCKET_NAME = "gradient-mrs-nifti"
DESTINATION_GCS_IMAGES_DIR = "01OCT2024"

# Filtering config.
MODALITY = "MR"
VALID_INSTITUTION_NAMES = ["tachyeres", "thryothor", "xenops"]
MIN_VOLUME_DEPTH = 10
MAX_VOLUME_DEPTH = 200

# DICOM.
DICOM_MODALITIES_MAPPING = {
    pydicom.uid.MRImageStorage: "MR"
}


def dump_config():
    return f"""Using the following configuration settings:
------------------------------------------------------
Display name: {DISPLAY_NAME}
Output dir: {OUTPUT_DIR}
Skip existing files: {SKIP_EXISTING_FILES}
Source GCS bucket name: {SOURCE_GCS_BUCKET_NAME}
Source GCS reports file: {SOURCE_GCS_REPORTS_FILE}
Source GCS images dir: {SOURCE_GCS_IMAGES_DIR}
Destination GCS bucket name: {DESTINATION_GCS_BUCKET_NAME}
Destination GCS images dir: {DESTINATION_GCS_IMAGES_DIR}
Modality: {MODALITY}
Valid institution names: {VALID_INSTITUTION_NAMES}
Min volume depth: {MIN_VOLUME_DEPTH}
Max volume depth: {MAX_VOLUME_DEPTH}
DICOM modalities mapping: {DICOM_MODALITIES_MAPPING}
------------------------------------------------------"""
