import argparse
import ast
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import pandas as pd
import pydicom
from PIL import Image
from tqdm import tqdm

from epsutils.dicom import dicom_utils
from epsutils.image import image_utils
from epsutils.logging import logging_utils


def filter_study_images(args):
    images_base_path, allowed_dicom_tag_values, df_row = args

    image_paths = ast.literal_eval(df_row["image_paths"])
    study_id = df_row["study_id"]
    patient_id = df_row["patient_id"]
    age = df_row["age"]
    study_date = df_row["study_date"]
    body_part = df_row["body_part"]
    gender = df_row["gender"]
    study_description = df_row["study_description"]
    manufacturer = df_row["manufacturer"]
    model_name = df_row["model_name"]

    filtered_image_paths = []

    for image_path in image_paths:
        try:
            dicom_file = pydicom.dcmread(os.path.join(images_base_path, image_path))

            # Check modality.

            if dicom_file.Modality not in allowed_dicom_tag_values["modalities"]:
                raise ValueError(f"Unsupported modality: {dicom_file.Modality}")

            # Check SOP class UID.

            if dicom_file.SOPClassUID not in allowed_dicom_tag_values["sop_class_uids"]:
                raise ValueError(f"Unsupported SOPClassUID: {pydicom.uid.UID(dicom_file.SOPClassUID).name}")

            # Check study instance UID.

            if dicom_file.StudyInstanceUID != study_id:
                raise ValueError(f"StudyInstanceUID mismatch: {dicom_file.StudyInstanceUID} (DICOM) != {study_id} (report)")

            # Check study date.

            if not dicom_utils.compare_dates(dicom_file.StudyDate, study_date):
                raise ValueError(f"StudyDate mismatch: {dicom_file.StudyDate} (DICOM) != {study_date} (report)")

            # Check study description.
            # TODO: Fix and uncomment.

            """
            if hasattr(dicom_file, "StudyDescription") and dicom_file.StudyDescription.upper() != study_description.upper():
                raise ValueError(f"StudyDescription mismatch: {dicom_file.StudyDescription} (DICOM) != {study_description} (report)")
            """

            # Check patient ID.

            if dicom_file.PatientID != patient_id:
                raise ValueError(f"PatientID mismatch: {dicom_file.PatientID} (DICOM) != {patient_id} (report)")

            # Check patient age.

            dicom_age = dicom_utils.age_to_years(dicom_file.PatientAge)
            report_age = dicom_utils.age_to_years(age)

            if dicom_age != report_age:
                raise ValueError(f"PatientAge mismatch: {dicom_file.PatientAge} (DICOM) != {age} (report)")

            # Check patient gender.
            # TODO: Fix and uncomment.

            """
            if dicom_file.PatientSex.upper() != gender.upper():
                raise ValueError(f"PatientSex mismatch: {dicom_file.PatientSex} (DICOM) != {gender} (report)")
            """

            # Check body part.
            # TODO: Fix and uncomment.

            """
            if dicom_file.BodyPartExamined.upper() != body_part.upper():
                raise ValueError(f"BodyPart mismatch: {dicom_file.BodyPartExamined} (DICOM) != {body_part} (report)")
            """

            # Check manufacturer.
            # TODO: Fix and uncomment.

            """
            if hasattr(dicom_file, "Manufacturer"):
                if str(dicom_file.Manufacturer).upper() != str(manufacturer).upper():
                    raise ValueError(f"Manufacturer mismatch: {dicom_file.Manufacturer} (DICOM) != {manufacturer} (report)")
            """

            # Check model.
            # TODO: Fix and uncomment.

            """
            if hasattr(dicom_file, "ManufacturerModelName"):
                if str(dicom_file.ManufacturerModelName).upper() != str(model_name).upper():
                    raise ValueError(f"Model mismatch: {dicom_file.ManufacturerModelName} (DICOM) != {model_name} (report)")
            """

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

            filtered_image_paths.append(image_path)

        except Exception as e:
            logging.warning(f"{str(e)} (patient ID: {patient_id}, study ID: {study_id}, image path: {image_path}")

    return filtered_image_paths


def filter_images(reports_file_with_image_paths, images_base_path, allowed_dicom_tag_values):
    print("Loading reports file")

    reports_df = pd.read_csv(reports_file_with_image_paths)

    if "image_paths" not in reports_df.columns:
        raise ValueError("Missing column 'image_paths' in the reports file")

    df_rows = reports_df.to_dict(orient="records")

    print("Image filtering started")

    with ProcessPoolExecutor() as executor:
        filtered_image_paths = list(tqdm(executor.map(filter_study_images, [(images_base_path, allowed_dicom_tag_values, row) for row in df_rows]),
                                         total=len(df_rows),
                                         desc="Processing"))

    assert len(filtered_image_paths) == len(reports_df)

    filtered_df = reports_df.copy()
    filtered_df["filtered_image_paths"] = filtered_image_paths

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
    reports_df = filter_images(reports_file_with_image_paths=args.reports_file_with_image_paths,
                               images_base_path=args.images_base_path,
                               allowed_dicom_tag_values=args.allowed_dicom_tag_values)

    # Save reports.
    save_reports(reports_df=reports_df, output_reports_file_path=args.output_reports_file_path)


if __name__ == "__main__":
    REPORTS_FILE_WITH_IMAGE_PATHS = "/mnt/efs/all-cxr/segmed/batch4/segmed_batch_4_merged_reports_with_image_paths.csv"
    IMAGES_BASE_PATH = "/mnt/efs/all-cxr/segmed/batch4"
    OUTPUT_REPORTS_FILE_PATH = "/mnt/efs/all-cxr/segmed/batch4/segmed_batch_4_merged_reports_with_image_paths_filtered.csv"

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

    args = argparse.Namespace(reports_file_with_image_paths=REPORTS_FILE_WITH_IMAGE_PATHS,
                              images_base_path=IMAGES_BASE_PATH,
                              allowed_dicom_tag_values=ALLOWED_DICOM_TAG_VALUES,
                              output_reports_file_path=OUTPUT_REPORTS_FILE_PATH)

    main(args)
