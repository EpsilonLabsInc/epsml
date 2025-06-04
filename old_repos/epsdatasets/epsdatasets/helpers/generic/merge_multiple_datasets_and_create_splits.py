import argparse

import pandas as pd
from sklearn.model_selection import train_test_split


def merge_datasets(datasets_info):
    print(f"Merging {len(datasets_info)} datasets")

    dfs = []

    for index, dataset_info in enumerate(datasets_info):
        dataset_name = dataset_info["dataset_name"]
        file_name = dataset_info["file_name"]
        report_text_column = dataset_info["report_text_column"]
        labels_column = dataset_info["labels_column"]
        image_paths_column = dataset_info["image_paths_column"]

        print(f"{index + 1}/{len(datasets_info)}")

        # Load dataset.
        print(f"Loading {dataset_name} dataset from {file_name}")
        df = pd.read_csv(file_name, low_memory=False)
        print(f"Dataset has {len(df)} rows")

        # Drop columns that won't be used.
        df = df[[report_text_column, labels_column, image_paths_column]]
        print(f"Using the following columns: {df.columns}")

        # Rename columns.
        df.rename(columns={report_text_column: "report_text", labels_column: "labels", image_paths_column: "image_paths"}, inplace=True)
        print(f"Column names after renaming: {df.columns}")

        dfs.append(df)

    # Merge datasets.
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"Dataset size after merging: {len(merged_df)}")

    return merged_df


def create_splits(df, seed):
    print("Creating splits")

    train_df, temp_df = train_test_split(df, test_size=0.1, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)

    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")

    return train_df, val_df, test_df


def save_splits(train_df, train_output_file_path, val_df, val_output_file_path, test_df, test_output_file_path):
    print("Saving training split")
    train_df.to_csv(train_output_file_path, index=False)

    print("Saving validation split")
    val_df.to_csv(val_output_file_path, index=False)

    print("Saving test split")
    test_df.to_csv(test_output_file_path, index=False)


def main(args):
    df = merge_datasets(datasets_info=args.datasets_info)

    train_df, val_df, test_df = create_splits(df=df, seed=args.seed)

    save_splits(train_df=train_df,
                train_output_file_path=args.train_output_file_path,
                val_df=val_df,
                val_output_file_path=args.val_output_file_path,
                test_df=test_df,
                test_output_file_path=args.test_output_file_path)


if __name__ == "__main__":
    DATASETS_INFO = [
        {
            "dataset_name": "gradient",
            "file_name": "/mnt/efs/all-cxr/gradient/GRADIENT_CR_ALL_BATCHES_with_uncertain_labels.csv",
            "dir_prefix_to_add": "gradient",
            "report_text_column": "report_text",  # TODO: Rename column.
            "labels_column": "labels",
            "image_paths_column": "image_paths"
        },
        {
            "dataset_name": "segmed_batch_1",
            "file_name": "/mnt/efs/all-cxr/segmed/batch1/segmed_batch_1_merged_reports_with_image_paths_filtered_standardized_mapped_modalities_with_uncertain_labels_cleaned_unflagged.csv",
            "dir_prefix_to_add": "segmed/batch1",
            "report_text_column": "cleaned_report_text",
            "labels_column": "labels",
            "image_paths_column": "filtered_image_paths"
        },
        {
            "dataset_name": "segmed_batch_2",
            "file_name": "/mnt/efs/all-cxr/segmed/batch2/segmed_batch_2_merged_reports_with_image_paths_filtered_standardized_mapped_modalities_with_uncertain_labels_cleaned_unflagged.csv",
            "dir_prefix_to_add": "segmed/batch2",
            "report_text_column": "cleaned_report_text",
            "labels_column": "labels",
            "image_paths_column": "filtered_image_paths"
        },
    ]

    TRAIN_OUTPUT_FILE_PATH = "gradient_batches_1-5_segmed_batches_1-2_simonmed_batches_1-10_reports_with_labels_train.csv"
    VAL_OUTPUT_FILE_PATH = "gradient_batches_1-5_segmed_batches_1-2_simonmed_batches_1-10_reports_with_labels_val.csv"
    TEST_OUTPUT_FILE_PATH = "gradient_batches_1-5_segmed_batches_1-2_simonmed_batches_1-10_reports_with_labels_test.csv"
    SEED = 42

    args = argparse.Namespace(datasets_info=DATASETS_INFO,
                              train_output_file_path=TRAIN_OUTPUT_FILE_PATH,
                              val_output_file_path=VAL_OUTPUT_FILE_PATH,
                              test_output_file_path=TEST_OUTPUT_FILE_PATH,
                              seed=SEED)

    main(args)
