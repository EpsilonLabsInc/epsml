from epsutils.gcs import gcs_utils

EPSILON_GCS_BUCKET_NAME = "gradient-crs"
EPSILON_GCS_DIR = "20DEC2024"


def main():
    # Get all files in the bucket.
    print(f"Getting a list of all the files in '{EPSILON_GCS_DIR}' dir of the '{EPSILON_GCS_BUCKET_NAME}' GCS bucket")
    all_files = gcs_utils.list_files(gcs_bucket_name=EPSILON_GCS_BUCKET_NAME, gcs_dir=EPSILON_GCS_DIR)
    print(f"All files found: {len(all_files)}")

    # Extract only TXT files.
    txt_files = [file for file in all_files if file.endswith(".txt")]
    print(f"All TXT files found: {len(txt_files)}")

    # Get distinct studies.
    distinct_studies = set()
    for txt_file in txt_files:
        study_dir = txt_file.split("_series")[0]
        distinct_studies.add(study_dir)
    print(f"Num distinct studies: {len(distinct_studies)}")


if __name__ == "__main__":
    main()
