import argparse

import pandas as pd

from epsutils.csv import csv_utils


def main(args):
    print(f"Subtracting {args.csv_to_subtract} from {args.csv_to_subtract_from}")
    original_df = pd.read_csv(args.csv_to_subtract_from, low_memory=False)
    rows_to_remove = pd.read_csv(args.csv_to_subtract, low_memory=False)
    df = csv_utils.subtract_matching_rows(original_df=original_df, rows_to_remove=rows_to_remove)

    print("Running sanity check")
    num_original = len(original_df)
    num_to_remove = len(rows_to_remove)
    num_result = len(df)
    res = (num_original == num_result + num_to_remove)
    if res:
        print("Sanity check successful")
    else:
        print("Sanity check failed")
        print(f"Num original: {num_original}")
        print(f"Num to remove: {num_to_remove}")
        print(f"Num result: {num_result}")
        print(f"Num result + num to remove: {num_result + num_to_remove}")

        key = input("Would you still like to proceed? (y/n): ")
        if key.strip().lower() != "y":
            print("Process aborted.")
            return

    print(f"Saving subtracted dataset to {args.output_file}")
    df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    CSV_TO_SUBTRACT_FROM = "/mnt/training/splits/gpt/spine/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_spine_val.csv"
    CSV_TO_SUBTRACT = "/mnt/training/splits/gpt/spine/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_strict_spine_val.csv"
    OUTPUT_FILE = "/mnt/training/splits/gpt/spine/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_non_strict_only_spine_val.csv"

    args = argparse.Namespace(csv_to_subtract_from=CSV_TO_SUBTRACT_FROM,
                              csv_to_subtract=CSV_TO_SUBTRACT,
                              output_file=OUTPUT_FILE)

    main(args)
