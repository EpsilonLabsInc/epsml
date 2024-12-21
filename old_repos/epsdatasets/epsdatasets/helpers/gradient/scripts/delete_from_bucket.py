from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm

from epsutils.gcs import gcs_utils

INPUT_FILE = "/home/andrej/work/epsdatasets/epsdatasets/helpers/gradient/scripts/gradient_mrs_01OCT2024_tachyeres.txt"
DELIMITER = ";"
FILE_NAME_COLUMN_INDEX = 0
EPSILON_GCS_BUCKET_NAME = "gradient-mrs-nifti"
EPSILON_GCS_DIR = ""


def delete_file(file_name):
    gcs_utils.delete_file(gcs_bucket_name=EPSILON_GCS_BUCKET_NAME, gcs_file_name=file_name)


def main():
    # Get a list of files to delete.
    df = pd.read_csv(INPUT_FILE, delimiter=DELIMITER, header=None)
    files_to_delete = df.iloc[:, FILE_NAME_COLUMN_INDEX]
    print("Files to delete:")
    print(files_to_delete)
    print(f"Total files to delete: {len(files_to_delete)}")

    # Delete files in parallel.
    with ProcessPoolExecutor() as executor:  # Use default number of workers.
        results = list(tqdm(executor.map(delete_file, files_to_delete), total=len(files_to_delete), desc="Deleting"))


if __name__ == "__main__":
    main()
