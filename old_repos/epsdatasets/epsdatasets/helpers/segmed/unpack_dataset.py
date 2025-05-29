import argparse
import ast
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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


def merge_reports_files(master_reports_file_path, dataset_root_dir, handle_missing_studies):
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

    # Handle missing studies.

    if len(master_df) != len(merged_df):
        print(f"WARNING: Lengths of master dataset ({len(master_df)}) and merged dataset ({len(merged_df)}) differ")

        master_dataset_ids = set(master_df["dataset_id"])
        partial_dataset_ids = set([int(os.path.basename(os.path.dirname(f))) for f in partial_reports_files])

        diff = master_dataset_ids - partial_dataset_ids
        print(f"Dataset IDs in the master dataset but missing in the merged dataset: {len(diff)}")

        diff = partial_dataset_ids- master_dataset_ids
        print(f"Dataset IDs in the merged dataset but missing in the master dataset: {len(diff)}")

        master_study_ids = set(master_df["study_id"])
        merged_study_ids = set(merged_df["study_id"])

        diff = master_study_ids - merged_study_ids
        print(f"Study IDs in the master dataset but missing in the merged dataset: {len(diff)}")

        diff = merged_study_ids - master_study_ids
        print(f"Study IDs in the merged dataset but missing in the master dataset: {len(diff)}")

        if handle_missing_studies:
            print("Fixing inconsistency")
            master_df = master_df[master_df["study_id"].isin(merged_study_ids)]
            merged_df = merged_df[merged_df["study_id"].isin(master_study_ids)]
        else:
            raise ValueError(f"Lengths of master and merged dataset differ")

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

    return master_df


def find_study_images(dataset_root_dir, df_row):
    dataset_id = str(df_row["dataset_id"])
    patient_id = df_row["patient_id"]
    study_id = str(df_row["study_id"])

    # ZIP file index can be either a single integer or a tuple,
    # indicating that study images were spread across multiple ZIP files.

    value = df_row["zip_file_index"]
    if isinstance(value, str):
        zip_file_index = ast.literal_eval(value)
    else:
        zip_file_index = value

    if not isinstance(zip_file_index, tuple):
        zip_file_index = (zip_file_index,)

    zip_file_index = (int(index) for index in zip_file_index)

    # Generate base dir by joining dataset root dir and dataset ID of the current study.

    base_dir = os.path.join(dataset_root_dir, dataset_id)

    # Find all subdirs of the base dir ending with "-{zip_file_index}"

    sub_dirs = []

    for index in zip_file_index:
        suffix = f"-{index}"
        sub_dirs.extend([str(d) for d in Path(base_dir).iterdir() if d.is_dir() and d.name.endswith(suffix)])

    # Append study_id to each of the subdirs.

    study_dirs = [os.path.join(sub_dir, study_id) for sub_dir in sub_dirs if os.path.exists(os.path.join(sub_dir, study_id))]

    if len(study_dirs) == 0:
        print(f"WARNING: No study dirs found in {base_dir} (patient ID: {patient_id}, study ID: {study_id}, ZIP file index: {zip_file_index})")
        return []

    # Find all the images in the study dirs.

    image_paths = []

    for study_dir in study_dirs:
        image_paths.extend([str(image_path) for image_path in Path(study_dir).rglob("*.dcm")])

    return image_paths


def map_image_paths(reports_df, dataset_root_dir, images_base_path):
    print("Mapping image paths")

    df_rows = reports_df.to_dict(orient="records")

    with ThreadPoolExecutor() as executor:
        image_paths = list(tqdm(executor.map(lambda row: find_study_images(dataset_root_dir, row), df_rows), total=len(df_rows), desc="Processing"))

    mapped_df = reports_df.copy()
    mapped_df["image_paths"] = image_paths
    mapped_df["image_paths"] = mapped_df["image_paths"].apply(lambda paths: [str(Path(path).relative_to(images_base_path)) for path in paths])

    return mapped_df


def save_reports(reports_df, output_reports_file_path):
    print(f"Saving reports to {output_reports_file_path}")
    reports_df.to_csv(output_reports_file_path, index=False)


def main(args):
    # Extract ZIP archives.
    if args.extract_zip_archives:
        zip_utils.extract_zip_archives(root_dir=args.dataset_root_dir,
                                       delete_after_extraction=args.delete_zip_archives_after_extraction)

    # Merge reports files.
    reports_df = merge_reports_files(master_reports_file_path=args.master_reports_file_path,
                                     dataset_root_dir=args.dataset_root_dir,
                                     handle_missing_studies=args.handle_missing_studies)

    # Map image paths.
    reports_df = map_image_paths(reports_df=reports_df,
                                 dataset_root_dir=args.dataset_root_dir,
                                 images_base_path=args.output_base_path)

    # Save reports.
    save_reports(reports_df=reports_df,
                 output_reports_file_path=args.output_reports_file_path)


if __name__ == "__main__":
    DATASET_ROOT_DIR = "/mnt/efs/all-cxr/segmed/batch2"
    MASTER_REPORTS_FILE_PATH = "/mnt/efs/all-cxr/segmed/batch2/CO2-658_part2.csv"
    EXTRACT_ZIP_ARCHIVES = True
    DELETE_ZIP_ARCHIVES_AFTER_EXTRACTION = True
    HANDLE_MISSING_STUDIES = True
    OUTPUT_REPORTS_FILE_PATH = "/mnt/efs/all-cxr/segmed/batch2/segmed_batch_2_merged_reports_with_image_paths.csv"
    OUTPUT_BASE_PATH = "/mnt/efs/all-cxr/segmed/batch2"

    args = argparse.Namespace(dataset_root_dir=DATASET_ROOT_DIR,
                              master_reports_file_path=MASTER_REPORTS_FILE_PATH,
                              extract_zip_archives=EXTRACT_ZIP_ARCHIVES,
                              delete_zip_archives_after_extraction=DELETE_ZIP_ARCHIVES_AFTER_EXTRACTION,
                              handle_missing_studies=HANDLE_MISSING_STUDIES,
                              output_reports_file_path=OUTPUT_REPORTS_FILE_PATH,
                              output_base_path=OUTPUT_BASE_PATH)

    main(args)
