import pydicom


# General config.
DISPLAY_NAME = "gradient-cr-20DEC2024"
OUTPUT_DIR = "./output"
SKIP_EXISTING_FILES = False

# GCS config.
SOURCE_GCS_BUCKET_NAME = "epsilon-data-us-central1"
SOURCE_GCS_REPORTS_FILE = "GRADIENT-DATABASE/CR/20DEC2024/epsilon-cr-export-18dec2024.csv"
SOURCE_GCS_IMAGES_DIR = "GRADIENT-DATABASE/CR/20DEC2024/deid"
DESTINATION_GCS_BUCKET_NAME = "gradient-crs"
DESTINATION_GCS_IMAGES_DIR = "20DEC2024"

# Filtering config.
MODALITIES = ["CR", "DX", "DR"]
VALID_INSTITUTION_NAMES = ["thryothor", "xenops"]
MIN_VOLUME_DEPTH = None
MAX_VOLUME_DEPTH = None

# DICOM.
DICOM_MODALITIES_MAPPING = {
    pydicom.uid.ComputedRadiographyImageStorage: "CR",
    pydicom.uid.DigitalXRayImageStorageForProcessing: "DX",
    pydicom.uid.DigitalXRayImageStorageForPresentation: "DX"
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
Modalities: {MODALITIES}
Valid institution names: {VALID_INSTITUTION_NAMES}
Min volume depth: {MIN_VOLUME_DEPTH}
Max volume depth: {MAX_VOLUME_DEPTH}
DICOM modalities mapping: {DICOM_MODALITIES_MAPPING}
------------------------------------------------------"""
