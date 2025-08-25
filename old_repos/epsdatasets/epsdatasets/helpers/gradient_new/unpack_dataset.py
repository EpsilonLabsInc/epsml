import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from epsutils.tar import tar_utils
from epsutils.misc import misc_utils


def merge_reports_file(reports_file_path, dicom_meta_data_file_path, handle_duplicates, handle_missing_studies):
    # Load reports file.
    print("Loading reports file")
    df1 = pd.read_csv(reports_file_path, low_memory=False)
    print(f"Reports file has {len(df1)} rows")

    # Handle duplicates.
    num_duplicates = df1["study_instance_uid"].duplicated().sum()
    if num_duplicates > 0:
        if handle_duplicates:
            print(f"Reports file has {num_duplicates} duplicates in study_instance_uid")
            df1 = df1.drop_duplicates(subset="study_instance_uid", keep="first")
            print(f"After removing duplicates reports file has {len(df1)} rows")
        else:
            raise ValueError(f"Reports file has {num_duplicates} duplicates in study_instance_uid")

    # Load DICOM meta data file.
    print("Loading DICOM meta data file")
    df2 = pd.read_csv(dicom_meta_data_file_path, low_memory=False)
    print(f"DICOM meta data file has {len(df2)} rows")

    # Rename columns to snake_case.
    print("Renaming DICOM meta data file columns to snake_case")
    df2.columns = [misc_utils.pascal_case_to_snake_case(col) for col in df2.columns]
    print(f"DICOM meta data file columns after renaming: {df2.columns.tolist()}")

    # Handle duplicates.
    num_duplicates = df2["study_instance_uid"].duplicated().sum()
    if num_duplicates > 0:
        if handle_duplicates:
            print(f"DICOM meta data file has {num_duplicates} duplicates in study_instance_uid")
            df2 = df2.drop_duplicates(subset="study_instance_uid", keep="first")
            print(f"After removing duplicates DICOM meta data file has {len(df2)} rows")
        else:
            raise ValueError(f"DICOM meta data file has {num_duplicates} duplicates in study_instance_uid")

    # Look for mismatches in study_instance_uid between both files.
    uids_only_in_df1 = df1[~df1["study_instance_uid"].isin(df2["study_instance_uid"])]
    print(f"Study instance UIDs only in reports file: {len(uids_only_in_df1)}")
    uids_only_in_df2 = df2[~df2["study_instance_uid"].isin(df1["study_instance_uid"])]
    print(f"Study instance UIDs only in DICOM meta data file: {len(uids_only_in_df2)}")

    if not handle_missing_studies and (len(uids_only_in_df1) > 0 or len(uids_only_in_df2) > 0):
        raise ValueError("Mismatch in study_instance_uid between reports file and DICOM meta data file")

    # Merge both files.
    merged_df = pd.merge(df1, df2, on="study_instance_uid", how="inner")
    print(f"Merged file has {len(merged_df)} rows")

    return merged_df


def find_study_images(dataset_root_dir, df_row):
    study_instance_uid = str(df_row["study_instance_uid"])
    base_dir = os.path.join(dataset_root_dir, study_instance_uid)
    image_paths = list(Path(base_dir).rglob("*.dcm"))
    return image_paths


def map_image_paths(reports_df, dataset_root_dir, images_base_path):
    print("Mapping image paths")

    df_rows = reports_df.to_dict(orient="records")

    with ThreadPoolExecutor() as executor:
        image_paths = list(tqdm(executor.map(lambda row: find_study_images(dataset_root_dir, row), df_rows), total=len(df_rows), desc="Processing"))

    assert len(image_paths) == len(reports_df), "Length of image paths does not match length of reports dataframe"

    mapped_df = reports_df.copy()
    mapped_df["image_paths"] = image_paths
    mapped_df["image_paths"] = mapped_df["image_paths"].apply(lambda paths: [str(Path(path).relative_to(images_base_path)) for path in paths])

    return mapped_df


def save_reports(reports_df, output_reports_file_path):
    print(f"Saving reports to {output_reports_file_path}")
    reports_df.to_csv(output_reports_file_path, index=False)


def main(args):
    # Extract TAR archives.
    if args.extract_tar_archives:
        tar_utils.extract_tar_archives(root_dir=args.dataset_root_dir,
                                       max_workers=args.num_workers,
                                       delete_after_extraction=args.delete_tar_archives_after_extraction)

    # Merge reports files.
    reports_df = merge_reports_file(reports_file_path=args.reports_file_path,
                                    dicom_meta_data_file_path=args.dicom_meta_data_file_path,
                                    handle_duplicates=args.handle_duplicates,
                                    handle_missing_studies=args.handle_missing_studies)

    # Map image paths.
    reports_df = map_image_paths(reports_df=reports_df,
                                 dataset_root_dir=args.dataset_root_dir,
                                 images_base_path=args.output_base_path)

    # Save reports.
    save_reports(reports_df=reports_df,
                 output_reports_file_path=args.output_reports_file_path)


if __name__ == "__main__":
    DATASET_ROOT_DIR = "/mnt/all-data/sfs-gradient-new/01JUL2025"
    REPORTS_FILE_PATH = "/mnt/all-data/reports/gradient-new/01JUL2025/reports000000000000.csv"
    DICOM_META_DATA_FILE_PATH = "/mnt/all-data/reports/gradient-new/01JUL2025/dicom_metadata000000000000.csv"
    NUM_WORKERS = 1  # > 1 is unsafe because multiple TARs can contain overlapping paths.
    EXTRACT_TAR_ARCHIVES = True
    DELETE_TAR_ARCHIVES_AFTER_EXTRACTION = True
    HANDLE_DUPLICATES = True
    HANDLE_MISSING_STUDIES = True
    OUTPUT_REPORTS_FILE_PATH = "/mnt/all-data/reports/gradient-new/01JUL2025/gradient_01JUL2025_merged_reports_with_image_paths.csv"
    OUTPUT_BASE_PATH = "/mnt/all-data/sfs-gradient-new/01JUL2025"

    args = argparse.Namespace(dataset_root_dir=DATASET_ROOT_DIR,
                              reports_file_path=REPORTS_FILE_PATH,
                              dicom_meta_data_file_path=DICOM_META_DATA_FILE_PATH,
                              num_workers=NUM_WORKERS,
                              extract_tar_archives=EXTRACT_TAR_ARCHIVES,
                              delete_tar_archives_after_extraction=DELETE_TAR_ARCHIVES_AFTER_EXTRACTION,
                              handle_duplicates=HANDLE_DUPLICATES,
                              handle_missing_studies=HANDLE_MISSING_STUDIES,
                              output_reports_file_path=OUTPUT_REPORTS_FILE_PATH,
                              output_base_path=OUTPUT_BASE_PATH)

    main(args)
