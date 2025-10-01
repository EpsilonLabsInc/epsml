import argparse

import pandas as pd


def main(args):
    print(f"Loading {args.input_csv_file}")
    df = pd.read_csv(args.input_csv_file, low_memory=False)

    values = df[args.column_name_for_filtering].unique()
    print("Unique values in the column for filtering BEFORE filtering:")
    print(values)

    print(f"Number of rows in the dataset BEFORE filtering: {len(df)}")

    print("Filtering...")
    if isinstance(args.content_for_filtering, list):
        df = df[df[args.column_name_for_filtering].isin(args.content_for_filtering)]
    else:
        df = df[df[args.column_name_for_filtering] == args.content_for_filtering]

    values = df[args.column_name_for_filtering].unique()
    print("Unique values in the column for filtering AFTER filtering:")
    print(values)

    print(f"Number of rows in the dataset AFTER filtering: {len(df)}")

    print(f"Saving filtered dataset to {args.output_csv_file}")
    df.to_csv(args.output_csv_file, index=False)


if __name__ == "__main__":
    INPUT_CSV_FILE = "/mnt/training/splits/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_with_all_extremity_segments_val.csv"
    OUTPUT_CSV_FILE = "/mnt/training/splits/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_with_all_extremity_segments_filtered_val.csv"
    COLUMN_NAME_FOR_FILTERING = "all_extremity_segments"
    CONTENT_FOR_FILTERING = ["Arm", "Hand", "Shoulder", "Leg", "Foot", "Ankle", "Knee", "Other"]  # Can be a single value or a list of values.

    args = argparse.Namespace(input_csv_file=INPUT_CSV_FILE,
                              output_csv_file=OUTPUT_CSV_FILE,
                              column_name_for_filtering=COLUMN_NAME_FOR_FILTERING,
                              content_for_filtering=CONTENT_FOR_FILTERING)

    main(args)
