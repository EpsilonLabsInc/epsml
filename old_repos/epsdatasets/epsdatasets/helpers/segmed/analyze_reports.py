import argparse
import re
from collections import defaultdict
from enum import Enum
from io import BytesIO

import pandas as pd
from tqdm import tqdm

from epsutils.aws import aws_s3_utils


class ProgramMode(Enum):
    GET_UNIQUE_REPORT_TEXT_KEYS = 1


def get_reports_file(aws_s3_reports_file):
    print("Downloading reports file")

    aws_s3_data = aws_s3_utils.split_aws_s3_uri(aws_s3_reports_file)
    content = aws_s3_utils.download_file_as_bytes(aws_s3_bucket_name=aws_s3_data["aws_s3_bucket_name"], aws_s3_file_name=aws_s3_data["aws_s3_path"])

    print("Reading reports file")

    df = pd.read_csv(BytesIO(content))

    return df


def get_unique_report_text_keys(aws_s3_reports_file):
    df = get_reports_file(aws_s3_reports_file)

    print("Reports file head:")
    print(df.head())

    print("Gathering unique report text keys")

    unique_keys = {}
    pattern = r"\b(\w+)\s*:"  # Extracts all the words before ':' (whether or not there are spaces) from the report text.

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        keys = re.findall(pattern, row["report"])
        keys = [key.upper() for key in keys]

        for key in keys:
            if key in unique_keys:
                unique_keys[key] += 1
            else:
                unique_keys[key] = 1

    # Sort unique keys.
    unique_keys = dict(sorted(unique_keys.items(), key=lambda item: item[1], reverse=True))

    # Merge singular and plural.
    merged_keys = defaultdict(int)
    for key, value in unique_keys.items():
        if (key.endswith("S") and key.rstrip("S") in unique_keys) or (not key.endswith("S") and (key + "S") in unique_keys):
            singular_key = key.rstrip("S")
            base_key = f"{singular_key}(S)"
        else:
            base_key = key
        merged_keys[base_key] += value

    unique_keys = merged_keys

    # Compute percentage.
    unique_keys = {key: f"{value} ({round((value / len(df)) * 100, 2)}%)" for key, value in unique_keys.items()}

    print("Unique keys:")
    for key, value in unique_keys.items():
        print(f"{key}: {value}")


def main(args):
    if args.program_mode == ProgramMode.GET_UNIQUE_REPORT_TEXT_KEYS:
        get_unique_report_text_keys(args.aws_s3_reports_file)

    else:
        raise ValueError("Not implemented")


if __name__ == "__main__":
    AWS_S3_REPORTS_FILE = "s3://epsilonlabs-segmed/batches/batch1/CO2_588_Batch_1_Part_1_delivered_studies.csv"
    PROGRAM_MODE = ProgramMode.GET_UNIQUE_REPORT_TEXT_KEYS

    args = argparse.Namespace(aws_s3_reports_file=AWS_S3_REPORTS_FILE,
                              program_mode=PROGRAM_MODE)

    main(args)
