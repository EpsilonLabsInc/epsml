import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pydicom
from tqdm import tqdm

from epsutils.logging import logging_utils


def parse_image(image_path):
    dicom_file = pydicom.dcmread(image_path, force=True)
    accession_number = str(dicom_file.AccessionNumber)
    logging.info(json.dumps({"accession_number": accession_number, "image_path": str(image_path)}))


def main(args):
    logging_utils.configure_logger(logger_file_name="all_available_simonmed_images.jsonl", show_logging_level=False)

    print(f"Searching for all the images within the studies directory {args.studies_dir}")
    image_paths = list(Path(args.studies_dir).rglob("*.dcm"))

    print("Reading images")
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(parse_image, image_paths), total=len(image_paths), desc="Processing"))


if __name__ == "__main__":
    STUDIES_DIR = "/mnt/efs/all-cxr/simonmed/images/422ca224-a9f2-4c64-bf7c-bb122ae2a7bb"

    args = argparse.Namespace(studies_dir=STUDIES_DIR)

    main(args)
