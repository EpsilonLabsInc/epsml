import argparse
import os

import pandas as pd
from tqdm import tqdm

from epsutils.sys import sys_utils
from epsutils.zip import zip_utils


MASTER_REPORTS_FILE_COLUMNS = ["dataset_id", "patient_id", "study_id", "age", "gender", "exam_date",
                               "modality", "body_part", "study_description", "report",
                               "source_location", "site", "state"]

PARTIAL_REPORTS_FILE_COLUMNS = ["Study ID", "Patient ID", "Report", "Patient Age", "Patient Sex",
                                "Modality", "Body Part", "Study Date", "Manufacturer", "Model Name",
                                "Site ID", "Source Location", "Status", "Metadata On Study Level",
                                "Metadata On Series Level", "Zip File Index"]

PARTIAL_REPORTS_FILE_COLUMNS_MAPPING = {
    "Study ID": "study_id",
    "Patient ID": "patient_id",
    "Report": "report",
    "Patient Age": "age",
    "Patient Sex": "gender",
    "Modality": "modality",
    "Body Part": "body_part",
    "Study Date": "study_date",
    "Manufacturer": "manufacturer",
    "Model Name": "model_name",
    "Site ID": "site",
    "Source Location": "source_location",
    "Status": "status",
    "Metadata On Study Level": "meta_data_on_study_level",
    "Metadata On Series Level": "meta_data_on_series_level",
    "Zip File Index": "zip_file_index"
}


def merge_reports_files(dataset_root_dir, master_reports_file_path, output_reports_file_path):
    # Load master reports file.

    print("Loading master reports file")

    master_df = pd.read_csv(master_reports_file_path)
    assert master_df.columns.tolist() == MASTER_REPORTS_FILE_COLUMNS

    # Find all partial reports files.

    print("Searching for partial reports files")

    partial_reports_files = []

    for foldername, subfolders, filenames in os.walk(dataset_root_dir):
        # Prevent further recursion.
        if sys_utils.compute_dir_depth(foldername, dataset_root_dir) >= 1:
            subfolders[:] = []

        # Only numeric folders hold actual DICOMs, so skip non-numeric ones.
        if not sys_utils.get_containing_dir(foldername).isdigit():
            continue

        partial_reports_files.extend(os.path.join(foldername, f) for f in filenames if f.endswith(".csv"))

    print(f"Found {len(partial_reports_files)} partial reports files")

    # Load partial reports files.

    print("Loading partial reports files")

    report_dfs = [pd.read_csv(f) for f in tqdm(partial_reports_files, desc="Processing", unit="file")]

    # Merge partial reports files.

    print("Merging partial reports files")

    merged_df = pd.concat(report_dfs, ignore_index=True)

    # Rename columns in merged reports file.

    assert merged_df.columns.tolist() == PARTIAL_REPORTS_FILE_COLUMNS

    for old_name, new_name in PARTIAL_REPORTS_FILE_COLUMNS_MAPPING.items():
        if old_name in merged_df.columns:
            merged_df.rename(columns={old_name: new_name}, inplace=True)

    # Add missing columns to the master reports file.

    missing_cols = set(merged_df.columns) - set(master_df.columns)

    for col in missing_cols:
        master_df[col] = pd.NA

    print (f"Added the following missing columns to the master reports file: {missing_cols}")

    # Sort by Study ID.

    master_df = master_df.sort_values(by="study_id")
    merged_df = merged_df.sort_values(by="study_id")

    # Sync master and merged reports files.

    print("Syncing the data")

    assert len(master_df) == len(merged_df)

    for (master_index, master_row), (merged_index, merged_row) in tqdm(zip(master_df.iterrows(), merged_df.iterrows()), total=len(master_df), desc="Processing", unit="row"):
        assert master_row["patient_id"] == merged_row["patient_id"] and master_row["study_id"] == merged_row["study_id"]

        for col in missing_cols:
            master_df.loc[master_index, col] = merged_row[col]

    # Save the output file.

    print("Saving the output file")

    master_df.to_csv(output_reports_file_path, index=False)


def main(args):
    # Merge reports files.
    merge_reports_files(dataset_root_dir=args.dataset_root_dir,
                        master_reports_file_path=args.master_reports_file_path,
                        output_reports_file_path=args.output_reports_file_path)

    # Extract ZIP archives.
    if args.extract_zip_archives:
        zip_utils.extract_zip_archives(root_dir=args.dataset_root_dir,
                                       delete_after_extraction=args.delete_zip_archives_after_extraction)
    else:
        print("Skipping extraction of ZIP archives")


if __name__ == "__main__":
    DATASET_ROOT_DIR = "/mnt/efs/all-cxr/segmed/batch1"
    MASTER_REPORTS_FILE_PATH = "/mnt/efs/all-cxr/segmed/batch1/CO2_354/CO2_588_Batch_1_Part_1_delivered_studies.csv"
    EXTRACT_ZIP_ARCHIVES = True
    DELETE_ZIP_ARCHIVES_AFTER_EXTRACTION = True
    OUTPUT_REPORTS_FILE_PATH = "/mnt/efs/all-cxr/segmed/batch1/segmed_batch_1_merged_reports.csv"

    args = argparse.Namespace(dataset_root_dir=DATASET_ROOT_DIR,
                              master_reports_file_path=MASTER_REPORTS_FILE_PATH,
                              extract_zip_archives=EXTRACT_ZIP_ARCHIVES,
                              delete_zip_archives_after_extraction=DELETE_ZIP_ARCHIVES_AFTER_EXTRACTION,
                              output_reports_file_path=OUTPUT_REPORTS_FILE_PATH)

    main(args)
