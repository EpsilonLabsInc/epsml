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

            accession_number = int(dicom_file.AccessionNumber)

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


def filter_images(reports_files, studies_dir, allowed_dicom_tag_values):
    if len(reports_files) < 1:
        raise ValueError(f"At least one reports file required, got {len(reports_files)}")

    print("Loading reports file(s)")

    dfs = [pd.read_csv(f) for f in tqdm(reports_files, desc="Processing", unit="file")]
    reports_df = pd.concat(dfs, ignore_index=True)
    reports_dict = dict(zip(reports_df.iloc[:, 0], reports_df.iloc[:, 1]))

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

    # Filter images.
    reports_df = filter_images(reports_files=args.reports_files,
                               studies_dir=args.studies_dir,
                               allowed_dicom_tag_values=args.allowed_dicom_tag_values)

    # Save reports.
    save_reports(reports_df=reports_df, output_reports_file_path=args.output_reports_file_path)


if __name__ == "__main__":
    REPORTS_FILES = ["/mnt/efs/all-cxr/simonmed/batch1/Steinberg_2020_20110_CR.csv"]
    STUDIES_DIR = "/mnt/efs/all-cxr/simonmed/batch1/422ca224-a9f2-4c64-bf7c-bb122ae2a7bb"
    OUTPUT_REPORTS_FILE_PATH = "/mnt/efs/all-cxr/simonmed/batch1/simonmed_batch_1_reports_with_image_paths_filtered.csv"

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

    args = argparse.Namespace(reports_files=REPORTS_FILES,
                              studies_dir=STUDIES_DIR,
                              allowed_dicom_tag_values=ALLOWED_DICOM_TAG_VALUES,
                              output_reports_file_path=OUTPUT_REPORTS_FILE_PATH)

    main(args)
