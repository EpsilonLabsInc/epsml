import os
from io import StringIO

import pandas as pd

from epsutils.gcs import gcs_utils

GRADIENT_GCS_BUCKET_NAME = "epsilon-data-us-central1"
GRADIENT_GCS_REPORTS_DIR = "GRADIENT-DATABASE/CR/13JAN2025-C1"
REPORTS_FILES = [
    "reports_20250104170630.csv-00000-of-00034.csv",
    "reports_20250104170630.csv-00001-of-00034.csv",
    "reports_20250104170630.csv-00002-of-00034.csv",
    "reports_20250104170630.csv-00003-of-00034.csv",
    "reports_20250104170630.csv-00004-of-00034.csv",
    "reports_20250104170630.csv-00005-of-00034.csv",
    "reports_20250104170630.csv-00006-of-00034.csv",
    "reports_20250104170630.csv-00007-of-00034.csv",
    "reports_20250104170630.csv-00008-of-00034.csv",
    "reports_20250104170630.csv-00009-of-00034.csv",
    "reports_20250104170630.csv-00010-of-00034.csv",
    "reports_20250104170630.csv-00011-of-00034.csv",
    "reports_20250104170630.csv-00012-of-00034.csv",
    "reports_20250104170630.csv-00013-of-00034.csv",
    "reports_20250104170630.csv-00014-of-00034.csv",
    "reports_20250104170630.csv-00015-of-00034.csv",
    "reports_20250104170630.csv-00016-of-00034.csv",
    "reports_20250104170630.csv-00017-of-00034.csv",
    "reports_20250104170630.csv-00018-of-00034.csv",
    "reports_20250104170630.csv-00019-of-00034.csv",
    "reports_20250104170630.csv-00020-of-00034.csv",
    "reports_20250104170630.csv-00021-of-00034.csv",
    "reports_20250104170630.csv-00022-of-00034.csv",
    "reports_20250104170630.csv-00023-of-00034.csv",
    "reports_20250104170630.csv-00024-of-00034.csv",
    "reports_20250104170630.csv-00025-of-00034.csv",
    "reports_20250104170630.csv-00026-of-00034.csv",
    "reports_20250104170630.csv-00027-of-00034.csv",
    "reports_20250104170630.csv-00028-of-00034.csv",
    "reports_20250104170630.csv-00029-of-00034.csv",
    "reports_20250104170630.csv-00030-of-00034.csv",
    "reports_20250104170630.csv-00031-of-00034.csv",
    "reports_20250104170630.csv-00032-of-00034.csv",
    "reports_20250104170630.csv-00033-of-00034.csv"
]
OUTPUT_FILE = "reports_20250104170630.csv"


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
