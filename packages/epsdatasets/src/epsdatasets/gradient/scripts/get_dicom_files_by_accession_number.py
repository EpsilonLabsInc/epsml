import os

from epsutils.gcs import gcs_utils

EPSILON_GCS_BUCKET_NAME = "gradient-crs"
EPSILON_GCS_IMAGES_DIR = "22JUL2024"
ACCESSION_NUMBERS = [
    "GRDN2W4WV9QFJE9E", "GRDNW7X3DWLGIASY", "GRDN55WXDJ9RVWZO", "GRDNO3X3309QPK63", "GRDNB99MUWAWFBQ0", "GRDNSFSDMJSNTZZV",
    "GRDNAAD2G16EOC5F", "GRDNYHEV834OBWGP", "GRDNAYTXFWVI6247", "GRDNJSJ3774MQMEY", "GRDNRM25XJSFXRAC", "GRDNDG6HDRST1KUL",
    "GRDNKUNCE8Z52T1C", "GRDNA38JS5FUPFOI", "GRDNADOZLQUK0I44", "GRDNHQJ4YU9V5E3T", "GRDNMPF5D5DMJ7VV"
]


def main():
    print(f"Getting a list of files from the '{EPSILON_GCS_BUCKET_NAME}' GCS bucket")
    files_in_bucket = gcs_utils.list_files(gcs_bucket_name=EPSILON_GCS_BUCKET_NAME, gcs_dir=EPSILON_GCS_IMAGES_DIR)
    files_in_bucket = [file for file in files_in_bucket if file.endswith(".txt")]
    files_in_bucket = set(files_in_bucket)  # Sets have average-time complexity of O(1) for lookups. In contrast, lists have an average-time complexity of O(n).

    print("Searching for provided accession numbers")
    for accession_number in ACCESSION_NUMBERS:
        print("")
        print("----------------------------------")
        print(f"Accession number: {accession_number}")
        print("----------------------------------")

        content = f"_{accession_number}_"

        for file in files_in_bucket:
            if content in file:
                dicom_file = os.path.basename(file).replace("_", "/").replace(".txt", ".dcm")
                print(dicom_file)


if __name__ == "__main__":
    main()
