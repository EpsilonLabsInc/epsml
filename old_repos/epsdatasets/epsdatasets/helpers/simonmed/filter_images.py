import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd
import pydicom
from PIL import Image
from tqdm import tqdm

from epsutils.dicom import dicom_compression_utils
from epsutils.dicom import dicom_utils
from epsutils.image import image_utils
from epsutils.logging import logging_utils

BATCHES = {
    1: [
        {"reports_file": "batch1/Steinberg_2020_20110_CR.csv", "header": False, "accession_number_column": 0, "report_text_column": 1}
    ],
    2: [
        {"reports_file": "batch2/Sills_2018_29055_CR.csv", "header": False, "accession_number_column": 0, "report_text_column": 1},
        {"reports_file": "batch2/Tranisi_2020_916_CR.csv", "header": False, "accession_number_column": 0, "report_text_column": 1}
    ],
    3: [
        {"reports_file": "batch3/Shiraj_2018_23091_CR.csv", "header": False, "accession_number_column": 0, "report_text_column": 1}
    ],
    4: [
        {"reports_file": "batch4/Rosellini_2018_15109_CR_sent.xlsx", "header": False, "accession_number_column": 0, "report_text_column": 1}
    ],
    5: [
        {"reports_file": "batch5/Reports Anonymized (1).xlsx", "header": True, "accession_number_column": "Accession Number", "report_text_column": "Report Text"}
    ],
    6: [
        {"reports_file": "batch6/Golshani_2018_18455_CR.csv", "header": False, "accession_number_column": 0, "report_text_column": 1}
    ],
    7: [
        {"reports_file": "batch7/Gerace_2018_6305_CR.csv", "header": False, "accession_number_column": 0, "report_text_column": 1},
        {"reports_file": "batch7/Sherry_2020_26586_CR.csv", "header": False, "accession_number_column": 0, "report_text_column": 1}
    ],
    8: [
        {"reports_file": "batch8/Cho_2018_13832_CR.csv", "header": False, "accession_number_column": 0, "report_text_column": 1}
    ],
    9: [
        {"reports_file": "batch9/Carlson_2018_13533_CR.csv", "header": False, "accession_number_column": 0, "report_text_column": 1},
        {"reports_file": "batch9/Leon_2018_10565_CR.csv", "header": False, "accession_number_column": 0, "report_text_column": 1},
        {"reports_file": "batch9/Sommerville_2018_9390_CR.csv", "header": False, "accession_number_column": 0, "report_text_column": 1}
    ],
    10: [
        {"reports_file": "batch10/Molitor_2023_38743.csv", "header": False, "accession_number_column": 0, "report_text_column": 1}
    ]
}


def filter_study_images(study_id, studies_dir, allowed_dicom_tag_values, reports_dict):

    study_dir = os.path.join(studies_dir, study_id)
    image_paths = list(Path(study_dir).rglob("*.dcm"))

    images = []

    for image_path in image_paths:
        try:
            try:
                dicom_file = pydicom.dcmread(image_path)
            except:
                dicom_file = pydicom.dcmread(image_path, force=True)
                dicom_file = dicom_compression_utils.handle_dicom_compression(dicom_file)

            accession_number = str(dicom_file.AccessionNumber)

            if accession_number not in reports_dict:
                return {}

            # Check modality.

            if dicom_file.Modality not in allowed_dicom_tag_values["modalities"]:
                raise ValueError(f"Unsupported modality: {dicom_file.Modality}")

            # Check SOP class UID.

            if dicom_file.SOPClassUID not in allowed_dicom_tag_values["sop_class_uids"]:
                raise ValueError(f"Unsupported SOPClassUID: {pydicom.uid.UID(dicom_file.SOPClassUID).name}")

            # Ignore non-primary/non-original and localizer image types.

            if hasattr(dicom_file, "ImageType"):
                if "PRIMARY" not in dicom_file.ImageType and "ORIGINAL" not in dicom_file.ImageType:
                    raise ValueError(f"Ignoring non-primary/non-original image type {dicom_file.ImageType}")

                if "LOCALIZER" in dicom_file.ImageType:
                    raise ValueError(f"Ignoring localizer image type {dicom_file.ImageType}")

            # Ignore multi-frame images.

            if hasattr(dicom_file, "NumberOfFrames") and dicom_file.NumberOfFrames != 1:
                raise ValueError(f"Multi-frame DICOM file with {dicom_file.NumberOfFrames} frames")

            # Ignore non-grayscale images.

            if dicom_file.SamplesPerPixel != 1:
                raise ValueError(f"Incorrect number of samples per pixel, should be 1 but got {dicom_file.SamplesPerPixel} instead")

            # Try to load the image.

            numpy_array = dicom_utils.get_dicom_image_from_dataset(dicom_file, {"window_width": 0, "window_center": 0})
            image = Image.fromarray(numpy_array)

            # Validate histogram.
            # TODO: Fix and uncomment.

            """
            config = image_utils.VALIDATE_IMAGE_HISTOGRAM_CONFIGURATIONS["CHEST_CR_SCAN" if dicom_file.BodyPartExamined.upper() == "CHEST" else "NON_CHEST_CR_SCAN"]
            res, err = image_utils.validate_image_histogram(image=image, config=config)

            if not res:
                raise ValueError(f"Histogram validation failed: {err}")
            """

            # Read DICOM data.

            image = {
                 "image_path": os.path.relpath(image_path, studies_dir),
                 "modality": dicom_file.Modality,
                 "accession_number": dicom_file.AccessionNumber,
                 "instance_id": dicom_file.SOPInstanceUID,
                 "study_id": dicom_file.StudyInstanceUID,
                 "series_id": dicom_file.SeriesInstanceUID,
                 "study_date": dicom_file.StudyDate,
                 "study_description": dicom_file.StudyDescription if hasattr(dicom_file, "StudyDescription") else "",
                 "patient_id": dicom_file.PatientID,
                 "age": dicom_file.PatientAge,
                 "gender": dicom_file.PatientSex,
                 "body_part": dicom_file.BodyPartExamined,
                 "manufacturer": dicom_file.Manufacturer if hasattr(dicom_file, "Manufacturer") else "",
                 "model_name": dicom_file.ManufacturerModelName if hasattr(dicom_file, "ManufacturerModelName") else "",
                 "report": reports_dict[accession_number]
            }

            images.append(image)

        except Exception as e:
            logging.warning(f"{str(e)} (image path: {image_path}")

    if len(images) == 0:
        return {}

    # Check data consistency.

    for image in images:
        try:
            # Check folder structure consistency.

            basename = os.path.splitext(os.path.basename(image["image_path"]))[0]

            if image["instance_id"] != basename:
                raise ValueError(f"SOPInstanceUID does not match the image file name: {image['instance_id']} <--> {basename}")

            if image["study_id"] != study_id:
                raise ValueError(f"StudyInstanceUID does not match study directory name: {image['study_id']} <--> {study_id}")

            parent = Path(image["image_path"]).parent.name

            if image["series_id"] != parent:
                raise ValueError(f"SeriesInstanceUID does not match series directory name: {image['series_id']} <--> {parent}")

            # Check DICOM data consistency.

            if image["modality"] != images[0]["modality"]:
                raise ValueError(f"Modality mismatch: {image['modality']} <--> {images[0]['modality']}")

            if image["accession_number"] != images[0]["accession_number"]:
                raise ValueError(f"AccessionNumber mismatch: {image['accession_number']} <--> {images[0]['accession_number']}")

            if image["study_id"] != images[0]["study_id"]:
                raise ValueError(f"StudyInstanceUID mismatch: {image['study_id']} <--> {images[0]['study_id']}")

            if image["study_date"] != images[0]["study_date"]:
                raise ValueError(f"StudyDate mismatch: {image['study_date']} <--> {images[0]['study_date']}")

            if image["study_description"] != images[0]["study_description"]:
                raise ValueError(f"StudyDescription mismatch: {image['study_description']} <--> {images[0]['study_description']}")

            if image["patient_id"] != images[0]["patient_id"]:
                raise ValueError(f"PatientID mismatch: {image['patient_id']} <--> {images[0]['patient_id']}")

            if image["age"] != images[0]["age"]:
                raise ValueError(f"PatientAge mismatch: {image['age']} <--> {images[0]['age']}")

            if image["gender"] != images[0]["gender"]:
                raise ValueError(f"PatientSex mismatch: {image['gender']} <--> {images[0]['gender']}")

            if image["body_part"] != images[0]["body_part"]:
                raise ValueError(f"BodyPartExamined mismatch: {image['body_part']} <--> {images[0]['body_part']}")

            if image["manufacturer"] != images[0]["manufacturer"]:
                raise ValueError(f"Manufacturer mismatch: {image['manufacturer']} <--> {images[0]['manufacturer']}")

            if image["model_name"] != images[0]["model_name"]:
                raise ValueError(f"ManufacturerModelName mismatch: {image['model_name']} <--> {images[0]['model_name']}")

        except Exception as e:
            logging.warning(f"Intra-study DICOM data consistency check failed: {str(e)} (study ID: {study_id}")
            return {}

    # Combine study data.

    study = {
        "accession_number": images[0]["accession_number"],
        "study_id": images[0]["study_id"],
        "study_date": images[0]["study_date"],
        "study_description": images[0]["study_description"],
        "modality": images[0]["modality"],
        "patient_id": images[0]["patient_id"],
        "age": images[0]["age"],
        "gender": images[0]["gender"],
        "body_part": images[0]["body_part"],
        "manufacturer": images[0]["manufacturer"],
        "model_name": images[0]["model_name"],
        "instance_ids": [image["instance_id"] for image in images],
        "series_ids": [image["series_id"] for image in images],
        "image_paths": [image["image_path"] for image in images],
        "report": images[0]["report"]
    }

    return study


def filter_images(batch_data, studies_dir, allowed_dicom_tag_values):
    assert len(batch_data) > 0

    print("Loading reports file(s)")

    dfs = []

    for index, item in enumerate(batch_data):
        reports_file = item["reports_file"]
        extension = os.path.splitext(reports_file)[1]
        header = 0 if item["header"] else None
        accession_number = item["accession_number_column"]
        report_text = item["report_text_column"]

        print(f"{index + 1}/{len(batch_data)} Loading reports file {reports_file}")

        if extension == ".csv":
            df = pd.read_csv(reports_file, header=header)
        elif extension == ".xlsx":
            df = pd.read_excel(reports_file, header=header, sheet_name=0)
        else:
            raise ValueError(f"Unknown reports file extension: {extension}")

        df = df[[accession_number, report_text]]
        cleaned_df = df.dropna(subset=[accession_number]).copy()
        cleaned_df[accession_number] = cleaned_df[accession_number].astype(str)
        dfs.append(cleaned_df)

        print(f"Dropped {len(df) - len(cleaned_df)} rows with NaN accession number out of total {len(df)} rows")

    reports_df = pd.concat(dfs, ignore_index=True)

    print(f"Reports dataset has {len(reports_df)} rows:")
    print(reports_df.head())

    reports_dict = dict(zip(reports_df.iloc[:, 0], reports_df.iloc[:, 1]))

    for key in list(reports_dict.keys()):
        try:
            float_value = float(key)
            int_value = int(float_value)
            new_key = str(int_value)
            reports_dict[new_key] = reports_dict.pop(key)
        except:
            pass

    print("Searching for all the studies within the studies directory")

    study_ids = [f.name for f in Path(studies_dir).iterdir() if f.is_dir()]

    print("Image filtering started")

    with ThreadPoolExecutor() as executor:
        studies = list(tqdm(executor.map(lambda study_id: filter_study_images(study_id, studies_dir, allowed_dicom_tag_values, reports_dict), study_ids),
                            total=len(study_ids),
                            desc="Processing"))

    assert len(studies) == len(study_ids)

    print("Generating filtered reports file")

    filtered_studies = [study for study in studies if study != {}]
    filtered_df = pd.DataFrame(filtered_studies)
    filtered_df.rename(columns={"image_paths": "filtered_image_paths"}, inplace=True)

    print(f"Filtered reports file contains {len(filtered_df)} out of {len(reports_df)} original studies")

    return filtered_df


def save_reports(reports_df, output_reports_file_path):
    print(f"Saving reports to {output_reports_file_path}")
    reports_df.to_csv(output_reports_file_path, index=False)


def main(args):
    # Configure logger.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging_utils.configure_logger(logger_file_name=f"{args.output_reports_file_path}_{timestamp}.log")

    # Suppress validation warnings.
    pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE

    # Get batch data.
    batch_data = BATCHES[args.batch_index]
    for item in batch_data:
        item["reports_file"] = os.path.join(args.batches_base_dir, item["reports_file"])

    # Filter images.
    reports_df = filter_images(batch_data=batch_data,
                               studies_dir=args.studies_dir,
                               allowed_dicom_tag_values=args.allowed_dicom_tag_values)

    # Save reports.
    save_reports(reports_df=reports_df, output_reports_file_path=args.output_reports_file_path)


if __name__ == "__main__":
    BATCH_INDEX = 5
    BATCHES_BASE_DIR = "/mnt/efs/all-cxr/simonmed/"
    STUDIES_DIR = "/mnt/efs/all-cxr/simonmed/images/422ca224-a9f2-4c64-bf7c-bb122ae2a7bb"
    OUTPUT_REPORTS_FILE_PATH = f"/mnt/efs/all-cxr/simonmed/batch{BATCH_INDEX}/simonmed_batch_{BATCH_INDEX}_reports_with_image_paths_filtered.csv"

    ALLOWED_DICOM_TAG_VALUES = {
        "modalities": [
            "CR",
            "DX"
        ],
        "sop_class_uids": [
            pydicom.uid.ComputedRadiographyImageStorage,
            pydicom.uid.DigitalXRayImageStorageForPresentation,
            pydicom.uid.DigitalXRayImageStorageForProcessing
        ]
    }

    args = argparse.Namespace(batch_index=BATCH_INDEX,
                              batches_base_dir=BATCHES_BASE_DIR,
                              studies_dir=STUDIES_DIR,
                              allowed_dicom_tag_values=ALLOWED_DICOM_TAG_VALUES,
                              output_reports_file_path=OUTPUT_REPORTS_FILE_PATH)

    main(args)
