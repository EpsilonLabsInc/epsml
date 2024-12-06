# General config.
DISPLAY_NAME = "fix-ct-16AGO2024"
OUTPUT_DIR = "/home/andrej/data/fix"

# GCS config.
GRADIENT_GCS_BUCKET_NAME = "epsilon-data-us-central1"
GRADIENT_GCS_IMAGES_DIR = "GRADIENT-DATABASE/CT/16AGO2024"
EPSILON_SOURCE_GCS_BUCKET_NAME = "gradient-cts-nifti"
EPSILON_SOURCE_GCS_IMAGES_DIR = "16AGO2024"
EPSILON_DESTINATION_GCS_BUCKET_NAME = "gradient-cts-fixed-nifti"
EPSILON_DESTINATION_GCS_IMAGES_DIR = "16AGO2024"


def dump_config():
    print(f"""Using the following configuration settings:
------------------------------------------------------
Display name: {DISPLAY_NAME}
Output dir: {OUTPUT_DIR}
Gradient GCS bucket name: {GRADIENT_GCS_BUCKET_NAME}
Gradient GCS images dir: {GRADIENT_GCS_IMAGES_DIR}
Epsilon source GCS bucket name: {EPSILON_SOURCE_GCS_BUCKET_NAME}
Epsilon source GCS images dir: {EPSILON_SOURCE_GCS_IMAGES_DIR}
Epsilon destination GCS bucket name: {EPSILON_DESTINATION_GCS_BUCKET_NAME}
Epsilon destination GCS images dir: {EPSILON_DESTINATION_GCS_IMAGES_DIR}
------------------------------------------------------""")
