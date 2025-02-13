import os
from io import StringIO

import pandas as pd

from epsutils.gcs import gcs_utils

CHEST_ONLY = True
GCS_CHEST_IMAGES_FILE = "gs://gradient-crs/archive/training/chest/chest_files_gradient_all_3_batches.csv"
GCS_DICOM_SERIES_DESCRIPTIONS_FILES = [
    "gs://gradient-crs/archive/series/gradient-crs-22JUL2024-series-descriptions.csv",
    "gs://gradient-crs/archive/series/gradient-crs-20DEC2024-series-descriptions.csv",
    "gs://gradient-crs/archive/series/gradient-crs-09JAN2025-series-descriptions.csv"
]


def main():
    if CHEST_ONLY:
        print("Downloading chest images file")
        gcs_data = gcs_utils.split_gcs_uri(GCS_CHEST_IMAGES_FILE)
        content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

        print("Generating a list of chest images")
        df = pd.read_csv(StringIO(content), header=None, sep=';')
        chest_images = set(df[0])

    # Get all DICOM series descriptions.
    descriptions = set()
    for gcs_file_name in GCS_DICOM_SERIES_DESCRIPTIONS_FILES:
        print(f"Downloading {gcs_file_name}")
        gcs_data = gcs_utils.split_gcs_uri(gcs_file_name)
        content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

        print("Extracting all DICOM series descriptions")
        lines = content.splitlines()
        for line in lines:
            description = line.split(";")[0].strip()
            file_name = line.split(";")[1].strip()

            if CHEST_ONLY and file_name not in chest_images:
                continue

            descriptions.add(description)

    print("All descriptions:")
    for description in descriptions:
        print(description)
    print("---")
    print(f"Total num of descriptions: {len(descriptions)}")
    print("")
    exit()

    # Get frontal descriptions only.
    frontal_descriptions = set()
    for d in descriptions:
        words = d.lower().split()
        if {"pa", "ap"}.intersection(words) and {"chest", "rib", "ribs"}.intersection(words) and not {"lateral", "lat"}.intersection(words):
            frontal_descriptions.add(d)

    print("Frontal descriptions:")
    print(frontal_descriptions)
    print(f"Total num of frontal descriptions: {len(frontal_descriptions)}")

if __name__ == "__main__":
    main()
