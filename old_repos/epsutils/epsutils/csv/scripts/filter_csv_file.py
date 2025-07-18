import argparse

import pandas as pd


def main(args):
    print(f"Loading {args.input_csv_file}")
    df = pd.read_csv(args.input_csv_file, low_memory=False)

    print(f"Number of rows in the dataset BEFORE filtering: {len(df)}")

    print("Filtering...")
    df = df[df[args.column_name_for_filtering] == args.content_for_filtering]

    print(f"Number of rows in the dataset AFTER filtering: {len(df)}")

    print(f"Saving filtered dataset to {args.output_csv_file}")
    df.to_csv(args.output_csv_file, index=False)


if __name__ == "__main__":
    INPUT_CSV_FILE = "/mnt/training/splits/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_with_arm_segment_val.csv"
    OUTPUT_CSV_FILE = "/mnt/training/splits/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_with_arm_segment_hand_val.csv"
    COLUMN_NAME_FOR_FILTERING = "arm_segment"
    CONTENT_FOR_FILTERING = "Hand"

    args = argparse.Namespace(input_csv_file=INPUT_CSV_FILE,
                              output_csv_file=OUTPUT_CSV_FILE,
                              column_name_for_filtering=COLUMN_NAME_FOR_FILTERING,
                              content_for_filtering=CONTENT_FOR_FILTERING)

    main(args)
