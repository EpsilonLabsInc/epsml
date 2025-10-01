import argparse

import pandas as pd
from tqdm import tqdm


def main(args):
    print(f"Loading source file {args.csv_to_get_cols_from}")
    src_df = pd.read_csv(args.csv_to_get_cols_from, low_memory=False)
    filtered_src_df = src_df[src_df[args.filter_col] == args.content_to_filter]

    print(f"Loading target file {args.csv_to_add_cols_to}")
    target_df = pd.read_csv(args.csv_to_add_cols_to, low_memory=False)

    for col in args.cols_to_get:
        target_df[col] = None

    print("Looking for matches and filling missing data")
    num_missing_rows = 0
    num_mismatches = 0
    for index, row in tqdm(target_df.iterrows(), total=len(target_df), desc="Processing"):
        pid = row["patient_id"]
        sid = row["study_uid"]

        match = filtered_src_df[(filtered_src_df["patient_id"] == pid) & (filtered_src_df["study_uid"] == sid)]

        if match.empty:
            # print(f"Unable to find row with patient_id = {pid} and study_uid = {sid} in the source dataset")
            num_missing_rows += 1
            if num_missing_rows % 100 == 0:
                print(f"Number of missing rows so far: {num_missing_rows}")
            continue

        if row["filtered_image_paths"] != match.iloc[0]["image_paths"]:
            print(f"Image paths mismatch")
            num_mismatches += 1
            continue

        for col in args.cols_to_get:
            target_df.at[index, col] = match.iloc[0][col]

    print(f"Number of missing rows: {num_missing_rows}")
    print(f"Number of mismatches: {num_mismatches}")

    print(f"Saving updated content to {args.csv_to_add_cols_to}")
    target_df.to_csv(args.csv_to_add_cols_to, index=False)


if __name__ == "__main__":
    CSV_TO_GET_COLS_FROM = "/mnt/training/splits/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_all.csv"
    CSV_TO_ADD_COLS_TO = "/mnt/sfs-simonmed/reports/batch1/simonmed_batch_1_reports_with_image_paths_filtered_standardized_with_dicom_data_mapped_modalities_mapped_body_parts_with_uncertain_labels_cleaned_unflagged.csv"
    FILTER_COL = "base_path"
    CONTENT_TO_FILTER = "simonmed"
    COLS_TO_GET = ["projection_classification", "chest_classification"]

    args = argparse.Namespace(csv_to_get_cols_from=CSV_TO_GET_COLS_FROM,
                              csv_to_add_cols_to=CSV_TO_ADD_COLS_TO,
                              filter_col=FILTER_COL,
                              content_to_filter=CONTENT_TO_FILTER,
                              cols_to_get=COLS_TO_GET)

    main(args)
