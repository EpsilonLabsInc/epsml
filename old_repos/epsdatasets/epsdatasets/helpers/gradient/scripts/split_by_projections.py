import os
from io import StringIO

import pandas as pd
from tqdm import tqdm

from epsutils.gcs import gcs_utils

GCS_CHEST_IMAGES_FILE = "gs://gradient-crs/archive/training/chest/chest_files_gradient_all_3_batches.csv"
GCS_ALL_PROJECTIONS_FILE = "gs://gradient-crs/archive/projections/gradient-crs-22JUL2024-all-projections.csv"
GCS_IMAGES_DIR = "GRADIENT-DATABASE/CR"
FRONTAL_OUTPUT_FILE = "gradient-crs-22JUL2024-chest-only-frontal-projections.csv"
LATERAL_OUTPUT_FILE = "gradient-crs-22JUL2024-chest-only-lateral-projections.csv"
OTHER_OUTPUT_FILE = "gradient-crs-22JUL2024-chest-only-other-projections.csv"


def main():
    # Download chest images file.
    print("Downloading chest images file")
    gcs_data = gcs_utils.split_gcs_uri(GCS_CHEST_IMAGES_FILE)
    content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

    # Generate a list of chest images.
    print("Generating a list of chest images")
    df = pd.read_csv(StringIO(content), header=None, sep=";")
    chest_images = set(df[0])

    # Download all projections file.
    print("Downloading all projections file")
    gcs_data = gcs_utils.split_gcs_uri(GCS_ALL_PROJECTIONS_FILE)
    content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

    # Generate a list of all projections.
    print("Generating a list of all projections")
    df = pd.read_csv(StringIO(content), header=None, sep=";")

    print(f"All projections: {len(df)}")

    # Remove non-chest projections.
    print("Removing non-chest projections")
    filtered_projections = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        file_name = os.path.join(GCS_IMAGES_DIR, row[0])
        if file_name not in chest_images:
            continue

        filtered_projections.append(row)

    print(f"All projections after non-chest removal: {len(filtered_projections)}")

    # Split by projections.
    print("Splitting by projections")
    frontal_projections = []
    lateral_projections = []
    other_projections = []
    for projection in filtered_projections:
        if projection[1] == "Frontal":
            frontal_projections.append(projection)
        elif projection[1] == "Lateral":
            lateral_projections.append(projection)
        elif projection[1] == "Other":
            other_projections.append(projection)
        else:
            raise ValueError(f"Unsupported projection {projection[1]}")

    frontal_projections = pd.DataFrame(frontal_projections)
    lateral_projections = pd.DataFrame(lateral_projections)
    other_projections = pd.DataFrame(other_projections)

    print(f"Frontal projections: {len(frontal_projections)}")
    print(f"Lateral projections: {len(lateral_projections)}")
    print(f"Other projections: {len(other_projections)}")

    # Save output files.
    print("Saving output files")
    frontal_projections.to_csv(FRONTAL_OUTPUT_FILE, header=False, index=False)
    lateral_projections.to_csv(LATERAL_OUTPUT_FILE, header=False, index=False)
    other_projections.to_csv(OTHER_OUTPUT_FILE, header=False, index=False)


if __name__ == "__main__":
    main()
