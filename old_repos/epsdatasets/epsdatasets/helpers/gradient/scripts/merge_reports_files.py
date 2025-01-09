import os
from io import StringIO

import pandas as pd

from epsutils.gcs import gcs_utils

GRADIENT_GCS_BUCKET_NAME = "epsilon-data-us-central1"
GRADIENT_GCS_REPORTS_DIR = "GRADIENT-DATABASE/CR/09JAN2025/original_unmerged_reports"
REPORTS_FILES = [
    "reports_20250108153205.csv-00000-of-00008.csv",
    "reports_20250108153205.csv-00001-of-00008.csv",
    "reports_20250108153205.csv-00002-of-00008.csv",
    "reports_20250108153205.csv-00003-of-00008.csv",
    "reports_20250108153205.csv-00004-of-00008.csv",
    "reports_20250108153205.csv-00005-of-00008.csv",
    "reports_20250108153205.csv-00006-of-00008.csv",
    "reports_20250108153205.csv-00007-of-00008.csv"
]
OUTPUT_FILE = "reports_20250108153205.csv"


def main():
    reports = []
    for reports_file in REPORTS_FILES:
        print(f"Downloading reports file {reports_file}")
        reports_file = os.path.join(GRADIENT_GCS_REPORTS_DIR, reports_file)
        content = gcs_utils.download_file_as_string(gcs_bucket_name=GRADIENT_GCS_BUCKET_NAME, gcs_file_name=reports_file)
        df = pd.read_csv(StringIO(content), sep=",", low_memory=False)
        print(f"Reports file has {len(df)} rows")
        reports.append(df)

    print("Merging reports files")
    merged_report = pd.concat(reports, ignore_index=True)

    print(f"Merged reports file has {len(merged_report)} rows")

    print("Saving merged reports file")
    merged_report.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
