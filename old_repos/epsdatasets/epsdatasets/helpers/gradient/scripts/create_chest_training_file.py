import csv
import os

import pandas as pd

CHEST_FILES_DIR = "c:/users/andrej/desktop/chest"
CHEST_FILE_PAIRS = [
    {"chest_non_chest_file": "gradient-crs-09JAN2025-chest_non_chest.csv", "dicom_tag_is_chest_file": "gradient-crs-09JAN2025-dicom_tag_is_chest.csv"},
    {"chest_non_chest_file": "gradient-crs-22JUL2024-chest_non_chest.csv", "dicom_tag_is_chest_file": "gradient-crs-22JUL2024-dicom_tag_is_chest.csv"}
]
OUTPUT_FILE = "chest_files_gradient_all_3_batches.csv"


def main():
    chest_files = []

    for index, pair in enumerate(CHEST_FILE_PAIRS):
        print(f"Processing {index + 1}/{len(CHEST_FILE_PAIRS)}")

        df = pd.read_csv(os.path.join(CHEST_FILES_DIR, pair["dicom_tag_is_chest_file"]), sep=";", header=None, low_memory=False)
        valid_dicom_files = set(df.iloc[:, 0])
        print(f"  Num DICOM files with BodyPartExamined == 'Chest': {len(df)}")

        df = pd.read_csv(os.path.join(CHEST_FILES_DIR, pair["chest_non_chest_file"]), sep=";", header=None, low_memory=False)
        df.columns = range(df.shape[1])
        print(f"  All chest/non-chest files: {len(df)}")
        dicom_files_to_add = df[df[1] == "CHEST"][0]
        print(f"  Num chest files: {len(dicom_files_to_add)}")

        for dicom_file in dicom_files_to_add:
            if dicom_file in valid_dicom_files:
                chest_files.append(dicom_file)

        print(f"  Output file has {len(chest_files)} rows")

    print(f"Writing to output file {OUTPUT_FILE}")
    with open(OUTPUT_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        for chest_file in chest_files:
            writer.writerow([chest_file])


if __name__ == "__main__":
    main()
