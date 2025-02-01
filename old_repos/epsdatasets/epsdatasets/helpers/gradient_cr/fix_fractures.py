import ast
import json
from io import StringIO

import pandas as pd
from tqdm import tqdm

from epsutils.gcs import gcs_utils

GCS_INPUT_FILE = "gs://epsilonlabs-filestore/cleaned_CRs/gradient_rm_bad_dcm_1211_nolabel.jsonl"
GCS_REPORTS_FILE_WITH_NEW_LABELS = "gs://epsilonlabs-filestore/cleaned_CRs/GRADIENT_CR_batch_1_new_fracture_labels.csv"
OUTPUT_FILE = "gradient_rm_bad_dcm_1211_nolabel_new_fractures.jsonl"
FRACTURE_LABEL = "Fracture"
REPORT_INCONSISTENCIES = False


def main():
    print("Downloading reports file with new labels")
    gcs_data = gcs_utils.split_gcs_uri(GCS_REPORTS_FILE_WITH_NEW_LABELS)
    content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

    print("Creating a dictionary with new labels")
    df = pd.read_csv(StringIO(content))
    new_labels_dict = dict(zip(df["accession_number"], df["labels"]))

    print("Downloading input file")
    gcs_data = gcs_utils.split_gcs_uri(GCS_INPUT_FILE)
    content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

    print("Fixing fractures in the input file")
    rows = content.splitlines()
    new_rows = []
    for row in tqdm(rows, total=len(rows), desc="Processing"):
        new_row = ast.literal_eval(row)
        accession_number = new_row["AssessionNumber"]
        assert accession_number in new_labels_dict
        new_labels = new_labels_dict[accession_number]

        try:
            new_labels = [label.strip() for label in new_labels.split(",")]
            assert len(new_labels) > 0
            new_labels.sort()
        except Exception as e:
            if REPORT_INCONSISTENCIES:
                print(f"Acc number: {accession_number}   New labels: {new_labels}   Error: {str(e)}")
            continue

        new_row["labels"] = new_labels
        new_rows.append(new_row)

    print("Writing fixed file to disk")
    with open(OUTPUT_FILE, "w") as file:
        for row in tqdm(new_rows, total=len(new_rows), desc="Writing"):
            json_line = json.dumps(row)
            file.write(json_line + "\n")


if __name__ == "__main__":
    main()
