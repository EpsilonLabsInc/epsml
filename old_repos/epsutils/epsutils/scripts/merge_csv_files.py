import argparse

import pandas as pd


def main(args):
    dfs = []
    total_len = 0

    for file_name in args.file_names:
        print(f"Loading {file_name}")
        df = pd.read_csv(file_name, low_memory=False)
        print(f"Number of rows in the dataset: {len(df)}")
        total_len += len(df)
        dfs.append(df)

    merged_df = pd.concat(dfs, axis=0)
    print(f"Number of rows in the merged dataset: {len(merged_df)}")

    assert total_len == len(merged_df)

    print(f"Saving merged dataset to {args.output_file}")
    merged_df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    FILE_NAMES = [
        "/mnt/efs/all-cxr/combined/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_train.csv",
        "/mnt/efs/all-cxr/combined/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_val.csv",
        "/mnt/efs/all-cxr/combined/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_test.csv"
    ]
    OUTPUT_FILE = "/mnt/efs/all-cxr/combined/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_all.csv"

    args = argparse.Namespace(file_names=FILE_NAMES,
                              output_file=OUTPUT_FILE)

    main(args)
