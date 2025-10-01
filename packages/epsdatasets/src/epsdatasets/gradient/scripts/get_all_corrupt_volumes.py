import ast
import os
import re
from io import StringIO

import pandas as pd
from tqdm import tqdm

from epsutils.gcs import gcs_utils

GCS_BUCKET_NAME = "gradient-cts-nifti"
GCS_REPORTS_FILE_NAME = "ct-16ago2024-batch-1-with_extra_columns_by_andrej.csv"
SLICES_CONSISTENCY_FILE = "/home/andrej/data/gradient-cts-fixed-nifti-16AGO2024-slices-consistency.txt"

def main():
    print(f"Getting a list of all corrupt files from the consistency file {SLICES_CONSISTENCY_FILE}")
    corrupt_files = set()

    with open(SLICES_CONSISTENCY_FILE, "r") as file:
        for line in file:
            match = re.search(r"(16AGO2024/.*?\.nii\.gz)", line)
            assert match
            corrupt_files.add(match.group(1))

    print(f"Num all corrupt files: {len(corrupt_files)}")

    print(f"Downloading reports file from the GCS bucket")
    reports_file_content = gcs_utils.download_file_as_string(gcs_bucket_name=GCS_BUCKET_NAME, gcs_file_name=GCS_REPORTS_FILE_NAME)

    print("Reading reports file")
    df = pd.read_csv(StringIO(reports_file_content), sep=",", low_memory=False)

    print("Searching for all corrupt volumes")
    all_corrupt_volumes = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        gcs_images_dir = row["GcsImagesDir"]
        primary_volumes = ast.literal_eval(row["PrimaryVolumes"])
        primary_volumes = [os.path.join(gcs_images_dir, primary_volume) for primary_volume in primary_volumes]

        for vol in primary_volumes:
            if vol in corrupt_files:
                all_corrupt_volumes.extend(primary_volumes)
                break

    print(f"All primary volumes: {len(all_corrupt_volumes)}")


if __name__ == "__main__":
    main()
