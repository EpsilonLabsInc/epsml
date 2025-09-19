import argparse
import ast

import pandas as pd
import tqdm as tq
from tqdm.auto import tqdm  # progress bars

tq.tqdm.pandas()  # enable pandas integration


from epsutils.labels.labels_by_body_part import (
    V1_3_0_FINEGRAINED_TO_CONSOLIDATED_BY_BODY_PART,
)


FINEGRAINED_TO_CONSOLIDATED = V1_3_0_FINEGRAINED_TO_CONSOLIDATED_BY_BODY_PART


def load_csv(file_name):
    # Read the CSV
    df = pd.read_csv(file_name)
    # Report original size
    print(f"Original rows: {df.shape[0]}")

    # Drop rows with NaN in 'relative_image_paths'
    df = df.dropna(subset=["relative_image_paths"])
    # Report filtered size
    print(f"Rows after dropping NaNs in 'relative_image_paths': {df.shape[0]}")

    df["relative_image_paths"] = df["relative_image_paths"].apply(ast.literal_eval)
    type_counts = (
        df["relative_image_paths"].apply(lambda v: type(v).__name__).value_counts()
    )
    print("Converted types in 'relative_image_paths':")
    print(type_counts)

    print("------------------" * 3)

    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Generate JSONL from labeled CSVs.")
    parser.add_argument(
        "--input_csv",
        type=str,
        help="Formatted reports pipeline CSV to remap.",
        required=True,
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Prefix for output .jsonl files (e.g., /home/eric/projects/all_data_cleaning/jsonl)",
    )

    return parser.parse_args()


def update_one_row(row):
    # TODO: remap labels
    return row


def main():

    args = parse_args()

    # Load CSVs
    print("Loading input csv")
    df = load_csv(args.input_csv)

    # Perform remapping.
    tqdm.pandas(desc="Remapping rows")
    updated_df = df.progress_apply(update_one_row, axis=1, result_type="expand")
    
    # Write output CSV.
    print("Writing output csv")
    updated_df.to_csv(args.output_csv, index=False)
    print(f"Saved output to {args.output_csv}")


if __name__ == "__main__":
    main()
