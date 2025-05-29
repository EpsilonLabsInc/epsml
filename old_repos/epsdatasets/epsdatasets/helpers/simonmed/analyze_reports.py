import argparse
import re
from collections import defaultdict
from enum import Enum

import pandas as pd
from tqdm import tqdm


class ProgramMode(Enum):
    GET_UNIQUE_REPORT_TEXT_KEYS = 1
    GET_REPORT_TEXT_EXAMPLES = 2


def get_unique_report_text_keys(reports_file):
    print("Loading reports file")

    df = pd.read_csv(reports_file, header=None)

    print("Reports file head:")
    print(df.head())

    print("Gathering unique report text keys")

    unique_keys = {}
    pattern = r"\b(\w+)\s*:"  # Extracts all the words before ':' (whether or not there are spaces) from the report text.

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        keys = re.findall(pattern, row[1])  # Second column corresponds to report text.
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


def get_report_text_examples(reports_file):
    print("Loading reports file")

    df = pd.read_csv(reports_file, header=None)
    df = df.iloc[:100][1]  # Second column corresponds to report text.
    out_file = "report_text_examples.csv"

    print(f"Saving report text examples to {out_file}")

    df.to_csv(out_file, index=False)


def main(args):
    if args.program_mode == ProgramMode.GET_UNIQUE_REPORT_TEXT_KEYS:
        get_unique_report_text_keys(args.reports_file)

    elif args.program_mode == ProgramMode.GET_REPORT_TEXT_EXAMPLES:
        get_report_text_examples(args.reports_file)

    else:
        raise ValueError("Not implemented")


if __name__ == "__main__":
    REPORTS_FILE = "/mnt/efs/all-cxr/simonmed/batch1/Steinberg_2020_20110_CR.csv"
    PROGRAM_MODE = ProgramMode.GET_UNIQUE_REPORT_TEXT_KEYS

    args = argparse.Namespace(reports_file=REPORTS_FILE,
                              program_mode=PROGRAM_MODE)

    main(args)
