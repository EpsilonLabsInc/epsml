import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def merge_datasets(datasets_info):
    print(f"Merging {len(datasets_info)} datasets")

    dfs = []

    for index, dataset_info in enumerate(datasets_info):
        dataset_name = dataset_info["dataset_name"]
        file_name = dataset_info["file_name"]
        images_base_path = dataset_info["images_base_path"]
        report_text_column = dataset_info["report_text_column"]
        labels_column = dataset_info["labels_column"]
        image_paths_column = dataset_info["image_paths_column"]

        print(f"{index + 1}/{len(datasets_info)}")

        # Load dataset.
        print(f"Loading {dataset_name} dataset from {file_name}")
        df = pd.read_csv(file_name, low_memory=False)
        print(f"Dataset has {len(df)} rows")

        # Populate base paths.
        print("Populating base paths")
        if dataset_name == "gradient":
            base_paths = df["batch"].apply(lambda batch_id: os.path.join(images_base_path, batch_id))
        else:
            base_paths = pd.Series([images_base_path] * len(df))
        assert len(base_paths) == len(df)
        df["base_path"] = base_paths

        # Drop columns that won't be used.
        columns_to_use = [
            "patient_id",
            "study_uid",
            "sex",
            "age",
            "body_part",
            report_text_column,
            labels_column,
            "structured_labels",
            image_paths_column,
            "base_path",
            "age_dicom",
            "sex_dicom",
            "body_part_dicom",
            "modality_dicom",
            "study_description_dicom",
            "chest_classification",
            "projection_classification"
        ]
        if dataset_name.startswith("segmed_"):
            columns_to_use.extend(["meta_data_on_study_level", "meta_data_on_series_level"])

        if dataset_name == "gradient" and not args.gradient_has_dicom_columns:
            columns_to_use.remove("age_dicom")
            columns_to_use.remove("sex_dicom")
            columns_to_use.remove("body_part_dicom")
            columns_to_use.remove("modality_dicom")
            columns_to_use.remove("study_description_dicom")

        df = df[columns_to_use]
        print(f"Using the following columns: {df.columns}")

        if dataset_name == "gradient" and not args.gradient_has_dicom_columns:
            df["age_dicom"] = None
            df["sex_dicom"] = None
            df["body_part_dicom"] = None
            df["modality_dicom"] = None
            df["study_description_dicom"] = None

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
            "file_name": "/mnt/all-data/reports/gradient/gradient_cr_all_batches_final.csv",
            "images_base_path": "gradient",
            "report_text_column": "cleaned_report_text",
            "labels_column": "labels",
            "image_paths_column": "relative_image_paths"
        },
        {
            "dataset_name": "segmed_batch_1",
            "file_name": "/mnt/all-data/reports/segmed/batch1/segmed_batch_1_final.csv",
            "images_base_path": "segmed/batch1",
            "report_text_column": "cleaned_report_text",
            "labels_column": "labels",
            "image_paths_column": "filtered_image_paths"
        },
        {
            "dataset_name": "segmed_batch_2",
            "file_name": "/mnt/all-data/reports/segmed/batch1/segmed_batch_2_final.csv",
            "images_base_path": "segmed/batch2",
            "report_text_column": "cleaned_report_text",
            "labels_column": "labels",
            "image_paths_column": "filtered_image_paths"
        },
        {
            "dataset_name": "segmed_batch_3",
            "file_name": "/mnt/all-data/reports/segmed/batch1/segmed_batch_3_final.csv",
            "images_base_path": "segmed/batch3",
            "report_text_column": "cleaned_report_text",
            "labels_column": "labels",
            "image_paths_column": "filtered_image_paths"
        },
        {
            "dataset_name": "segmed_batch_4",
            "file_name": "/mnt/all-data/reports/segmed/batch1/segmed_batch_4_final.csv",
            "images_base_path": "segmed/batch4",
            "report_text_column": "cleaned_report_text",
            "labels_column": "labels",
            "image_paths_column": "filtered_image_paths"
        },
        {
            "dataset_name": "segmed_batch_5",
            "file_name": "/mnt/all-data/reports/segmed/batch1/segmed_batch_5_final.csv",
            "images_base_path": "segmed/batch5",
            "report_text_column": "cleaned_report_text",
            "labels_column": "labels",
            "image_paths_column": "filtered_image_paths"
        },
        {
            "dataset_name": "segmed_batch_6",
            "file_name": "/mnt/all-data/reports/segmed/batch1/segmed_batch_6_final.csv",
            "images_base_path": "segmed/batch6",
            "report_text_column": "cleaned_report_text",
            "labels_column": "labels",
            "image_paths_column": "filtered_image_paths"
        },
        {
            "dataset_name": "segmed_batch_7",
            "file_name": "/mnt/all-data/reports/segmed/batch1/segmed_batch_7_final.csv",
            "images_base_path": "segmed/batch7",
            "report_text_column": "cleaned_report_text",
            "labels_column": "labels",
            "image_paths_column": "filtered_image_paths"
        },
        {
            "dataset_name": "segmed_batch_8",
            "file_name": "/mnt/all-data/reports/segmed/batch1/segmed_batch_8_final.csv",
            "images_base_path": "segmed/batch8",
            "report_text_column": "cleaned_report_text",
            "labels_column": "labels",
            "image_paths_column": "filtered_image_paths"
        },
        {
            "dataset_name": "segmed_batch_9",
            "file_name": "/mnt/all-data/reports/segmed/batch1/segmed_batch_9_final.csv",
            "images_base_path": "segmed/batch9",
            "report_text_column": "cleaned_report_text",
            "labels_column": "labels",
            "image_paths_column": "filtered_image_paths"
        },
        {
            "dataset_name": "simonmed_batch_1",
            "file_name": "/mnt/all-data/reports/simonmed/batch1/simonmed_batch_1_final.csv",
            "images_base_path": "simonmed",
            "report_text_column": "cleaned_report_text",
            "labels_column": "labels",
            "image_paths_column": "filtered_image_paths"
        },
        {
            "dataset_name": "simonmed_batch_2",
            "file_name": "/mnt/all-data/reports/simonmed/batch1/simonmed_batch_2_final.csv",
            "images_base_path": "simonmed",
            "report_text_column": "cleaned_report_text",
            "labels_column": "labels",
            "image_paths_column": "filtered_image_paths"
        },
        {
            "dataset_name": "simonmed_batch_3",
            "file_name": "/mnt/all-data/reports/simonmed/batch1/simonmed_batch_3_final.csv",
            "images_base_path": "simonmed",
            "report_text_column": "cleaned_report_text",
            "labels_column": "labels",
            "image_paths_column": "filtered_image_paths"
        },
        {
            "dataset_name": "simonmed_batch_4",
            "file_name": "/mnt/all-data/reports/simonmed/batch1/simonmed_batch_4_final.csv",
            "images_base_path": "simonmed",
            "report_text_column": "cleaned_report_text",
            "labels_column": "labels",
            "image_paths_column": "filtered_image_paths"
        },
        {
            "dataset_name": "simonmed_batch_5",
            "file_name": "/mnt/all-data/reports/simonmed/batch1/simonmed_batch_5_final.csv",
            "images_base_path": "simonmed",
            "report_text_column": "cleaned_report_text",
            "labels_column": "labels",
            "image_paths_column": "filtered_image_paths"
        },
        {
            "dataset_name": "simonmed_batch_6",
            "file_name": "/mnt/all-data/reports/simonmed/batch1/simonmed_batch_6_final.csv",
            "images_base_path": "simonmed",
            "report_text_column": "cleaned_report_text",
            "labels_column": "labels",
            "image_paths_column": "filtered_image_paths"
        },
        {
            "dataset_name": "simonmed_batch_7",
            "file_name": "/mnt/all-data/reports/simonmed/batch1/simonmed_batch_7_final.csv",
            "images_base_path": "simonmed",
            "report_text_column": "cleaned_report_text",
            "labels_column": "labels",
            "image_paths_column": "filtered_image_paths"
        },
        {
            "dataset_name": "simonmed_batch_8",
            "file_name": "/mnt/all-data/reports/simonmed/batch1/simonmed_batch_8_final.csv",
            "images_base_path": "simonmed",
            "report_text_column": "cleaned_report_text",
            "labels_column": "labels",
            "image_paths_column": "filtered_image_paths"
        },
        {
            "dataset_name": "simonmed_batch_9",
            "file_name": "/mnt/all-data/reports/simonmed/batch1/simonmed_batch_9_final.csv",
            "images_base_path": "simonmed",
            "report_text_column": "cleaned_report_text",
            "labels_column": "labels",
            "image_paths_column": "filtered_image_paths"
        },
        {
            "dataset_name": "simonmed_batch_10",
            "file_name": "/mnt/sfs-simonmed/reports/batch10/simonmed_batch_10_reports_with_image_paths_filtered_standardized_with_dicom_data_mapped_modalities_mapped_body_parts_with_uncertain_labels_cleaned_unflagged.csv",
            "images_base_path": "simonmed",
            "report_text_column": "cleaned_report_text",
            "labels_column": "labels",
            "image_paths_column": "filtered_image_paths"
        }
    ]

    TRAIN_OUTPUT_FILE_PATH = "gradient_batches_1-5_segmed_batches_1-9_simonmed_batches_1-10_reports_with_labels_train.csv"
    VAL_OUTPUT_FILE_PATH = "gradient_batches_1-5_segmed_batches_1-9_simonmed_batches_1-10_reports_with_labels_val.csv"
    TEST_OUTPUT_FILE_PATH = "gradient_batches_1-5_segmed_batches_1-9_simonmed_batches_1-10_reports_with_labels_test.csv"
    GRADIENT_HAS_DICOM_COLUMNS = False
    SEED = 42

    args = argparse.Namespace(datasets_info=DATASETS_INFO,
                              train_output_file_path=TRAIN_OUTPUT_FILE_PATH,
                              val_output_file_path=VAL_OUTPUT_FILE_PATH,
                              test_output_file_path=TEST_OUTPUT_FILE_PATH,
                              gradient_has_dicom_columns=GRADIENT_HAS_DICOM_COLUMNS,
                              seed=SEED)

    main(args)
