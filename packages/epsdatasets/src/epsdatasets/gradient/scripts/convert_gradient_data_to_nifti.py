import argparse
import ast

import pandas as pd

from epsutils.nifti import nifti_utils


def main():
    parser = argparse.ArgumentParser(description="Tool for converting Gradient DICOM volumes to NIfTI volumes")
    parser.add_argument("generated_data_file", help="Generated data file.")
    parser.add_argument("base_dir", help="Base directory to be prepended to the path in the generated data file. If it is None, prepending is not performed.")
    parser.add_argument("output_dir", help="Either local output directory (if gcs_bucket_name is None) or output directory in the GCS bucket (if gcs_bucket_name is not None).")
    parser.add_argument("gcs_bucket_name", help="Name of the GCS bucket to upload NIfTI files to.")
    parser.add_argument("new_generated_data_file", help="New generated data file.")
    parser.add_argument('--sanity_check', help="Perform sanity check.", action="store_true")

    args = parser.parse_args()
    generated_data_file = args.generated_data_file
    base_dir = None if args.base_dir == "" else args.base_dir
    output_dir = args.output_dir
    gcs_bucket_name = None if args.gcs_bucket_name == "" else args.gcs_bucket_name
    new_generated_data_file = args.new_generated_data_file
    sanity_check = args.sanity_check

    print(f"Loading generated data from '{generated_data_file}'")
    df = pd.read_csv(generated_data_file)
    df = df.map(ast.literal_eval)  # Make sure all the elements are converted from strings back to original Python types.

    print("Generating structured DICOM files")
    volumes = df["volume"].to_dict()
    structured_dicom_files = [volume for volume in volumes.values()]

    print("Closing generated data to save memory")
    del df

    print("Conversion started")

    if sanity_check:
        print("Sanity check will be performed")
    else:
        print("Sanity check won't be performed")

    nifti_utils.structured_dicom_files_to_nifti_files(
        structured_dicom_files=structured_dicom_files, base_dir=base_dir, output_dir=output_dir,
        gcs_bucket_name=gcs_bucket_name, max_workers=15, perform_sanity_check=sanity_check)

    print(f"Reloading generated data from '{generated_data_file}'")
    df = pd.read_csv(generated_data_file)
    df = df.map(ast.literal_eval)  # Make sure all the elements are converted from strings back to original Python types.

    print("Creating new generated data file")
    df["volume"] = df["volume"].apply(lambda volume: {"nifti_file": volume["path"].replace('/', '_') + ".nii.gz", "num_slices": len(volume["dicom_files"])})
    df.to_csv(new_generated_data_file, index=False)

    print("Finished")


if __name__ == "__main__":
    main()
