import argparse

import pandas as pd


def main(args):
    print(f"Loading {args.input_csv_file}")
    df = pd.read_csv(args.input_csv_file, low_memory=False)

    if args.new_column_name in df.columns:
        print(f"Deleting existing column '{args.new_column_name}'")
        df.drop(columns=[args.new_column_name], inplace=True)

    print(f"Renaming column '{args.old_column_name}' to '{args.new_column_name}'")
    df.rename(columns={args.old_column_name: args.new_column_name}, inplace=True)

    print(f"Saving updated dataset to {args.input_csv_file}")
    df.to_csv(args.input_csv_file, index=False)


if __name__ == "__main__":
    INPUT_CSV_FILE = "/mnt/all-data/reports/segmed/batch1/segmed_batch_1_merged_reports_with_image_paths_filtered.csv"
    OLD_COLUMN_NAME = "filtered_image_paths"
    NEW_COLUMN_NAME = "image_paths"

    args = argparse.Namespace(
        input_csv_file=INPUT_CSV_FILE,
        old_column_name=OLD_COLUMN_NAME,
        new_column_name=NEW_COLUMN_NAME
    )

    main(args)
