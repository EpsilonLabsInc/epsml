import ast
import logging
import math
import os
import warnings
from io import BytesIO

import pydicom
import SimpleITK as sitk
from google.cloud import storage
from PIL import Image

from epsdatasets.helpers.gradient import gradient_utils
from epsutils.dicom import dicom_utils
from epsutils.gcs import gcs_utils
from epsutils.image import image_utils
from epsutils.nifti import nifti_utils

import config


def process_row(row):
    try:
        process_row_impl(row)
    except Exception as e:
        print(f"Exception in process row impl: {str(e)}")


def process_row_impl(row):
    row_data = get_row_data(row)
    if row_data is None:
        logging.warning(f"Unable to parse row, study rejected (row ID: N/A)")
        return

    # Check institution name.
    if not any(valid_institution in row_data["institution_name"].lower() for valid_institution in config.VALID_INSTITUTION_NAMES):
        logging.warning(f"Invalid institution {row_data['institution_name']}, study rejected (row ID: {row_data['row_id']})")
        return

    # Check modality.
    if not any(modality in config.MODALITIES for modality in row_data["modalities"]):
        logging.warning(f"Invalid modality {row_data['modalities']}, study rejected (row ID: {row_data['row_id']})")
        return

    # Iterate volumes.
    for series_instance_uid in row_data["series_instance_uids"]:
        volume_dir = gradient_utils.get_gradient_instances_path(patient_id=row_data["patient_id"],
                                                                accession_number=row_data["accession_number"],
                                                                study_instance_uid=row_data["study_instance_uid"],
                                                                series_instance_uid=series_instance_uid)

        # Check if NIfTI and volume info files already exist in the destination GCS bucket.
        nifti_file_name = gradient_utils.gradient_instances_path_to_nifti_file_name(volume_dir)
        volume_info_file_name = gradient_utils.gradient_instances_path_to_volume_info_file_name(volume_dir)
        files_to_check = [
            os.path.join(config.DESTINATION_GCS_IMAGES_DIR, nifti_file_name),
            os.path.join(config.DESTINATION_GCS_IMAGES_DIR, volume_info_file_name)
        ]
        if config.SKIP_EXISTING_FILES and gcs_utils.check_if_files_exist(gcs_bucket_name=config.DESTINATION_GCS_BUCKET_NAME, gcs_file_names=files_to_check):
            continue

        # Get DICOM files from GCS bucket.
        dicom_files, err_msg = get_dicom_files_from_gcs(gcs_bucket_name=config.SOURCE_GCS_BUCKET_NAME, gcs_dir=os.path.join(config.SOURCE_GCS_IMAGES_DIR, volume_dir))
        if dicom_files is None:
            logging.warning(f"{err_msg}, series rejected (row ID: {row_data['row_id']})")
            continue

        # Sort DICOM files by the InstanceNumber.
        dicom_files = sort_dicom_files(dicom_files)
        if dicom_files is None:
            logging.warning(f"Error sorting DICOM files, series rejected (row ID: {row_data['row_id']})")
            continue

        # Validate DICOM files.
        res, err_msg = validate_dicom_files(dicom_files, row_data)
        if not res:
            logging.warning(f"{err_msg}, series rejected (row ID: {row_data['row_id']})")
            continue

        # Create NIfTI volume.
        volume, err_msg = create_nifti_volume(dicom_files)
        if volume is None:
            logging.warning(f"{err_msg}, series rejected (row ID: {row_data['row_id']})")
            continue

        # Create volume info.
        dicom_content = "\n".join(dicom_utils.read_all_dicom_tags_from_dataset(dicom_files[0]))
        volume_info = (
            f"Number of slices: {len(dicom_files)}\n"
            f"DICOM content:\n"
            f"{dicom_content}"
        )

        # Upload NIfTI volume.
        res, err_msg = upload_nifti_volume(volume=volume, volume_info=volume_info, nifti_file_name=nifti_file_name)
        if not res:
            logging.warning(f"{err_msg}, series rejected (row ID: {row_data['row_id']})")
            continue


def process_row_cr(row):
    try:
        process_row_cr_impl(row)
    except Exception as e:
        print(f"Exception in process row CR impl: {str(e)}")


def process_row_cr_impl(row):
    row_data = get_row_data(row)
    if row_data is None:
        logging.warning(f"Unable to parse row, study rejected (row ID: N/A)")
        return

    # Check institution name.
    if not any(valid_institution in row_data["institution_name"].lower() for valid_institution in config.VALID_INSTITUTION_NAMES):
        logging.warning(f"Invalid institution {row_data['institution_name']}, study rejected (row ID: {row_data['row_id']})")
        return

    # Check modality.
    if not any(modality in config.MODALITIES for modality in row_data["modalities"]):
        logging.warning(f"Invalid modality {row_data['modalities']}, study rejected (row ID: {row_data['row_id']})")
        return

    # Iterate volumes.
    for series_instance_uid in row_data["series_instance_uids"]:
        instances_dir = gradient_utils.get_gradient_instances_path(patient_id=row_data["patient_id"],
                                                                   accession_number=row_data["accession_number"],
                                                                   study_instance_uid=row_data["study_instance_uid"],
                                                                   series_instance_uid=series_instance_uid)

        # Get DICOM files from GCS bucket.
        dicom_files, err_msg = get_dicom_files_from_gcs(gcs_bucket_name=config.SOURCE_GCS_BUCKET_NAME, gcs_dir=os.path.join(config.SOURCE_GCS_IMAGES_DIR, instances_dir))
        if dicom_files is None:
            logging.warning(f"{err_msg}, series {series_instance_uid} rejected (row ID: {row_data['row_id']})")
            continue

        # Validate DICOM files.
        for dicom_file in dicom_files:
            if pydicom.tag.Tag("ImageType") not in dicom_file:
                logging.warning(f"ImageType tag not present in the DICOM file, DICOM file with SOP instance UID {dicom_file.SOPInstanceUID} in series {series_instance_uid} rejected (row ID: {row_data['row_id']})")
                continue

            if "PRIMARY" not in dicom_file.ImageType and "ORIGINAL" not in dicom_file.ImageType:
                logging.warning(f"Ignoring non-primary/non-original image type {dicom_file.ImageType}, DICOM file with SOP instance UID {dicom_file.SOPInstanceUID} in series {series_instance_uid} rejected (row ID: {row_data['row_id']})")
                continue

            if "LOCALIZER" in dicom_file.ImageType:
                logging.warning(f"Ignoring localizer image type {dicom_file.ImageType}, DICOM file with SOP instance UID {dicom_file.SOPInstanceUID} in series {series_instance_uid} rejected (row ID: {row_data['row_id']})")
                continue

            if dicom_file.SOPClassUID not in config.DICOM_MODALITIES_MAPPING:
                logging.warning(f"Unsupported SOP Class UID {pydicom.uid.UID(dicom_file.SOPClassUID).name}, DICOM file with SOP instance UID {dicom_file.SOPInstanceUID} in series {series_instance_uid} rejected (row ID: {row_data['row_id']})")
                continue

            if config.DICOM_MODALITIES_MAPPING[dicom_file.SOPClassUID] not in config.MODALITIES:
                logging.warning(f"Incorrect SOP Class UID {pydicom.uid.UID(dicom_file.SOPClassUID).name}, DICOM file with SOP instance UID {dicom_file.SOPInstanceUID} in series {series_instance_uid} rejected (row ID: {row_data['row_id']})")
                continue

            if dicom_file.Modality not in config.MODALITIES:
                logging.warning(f"Incorrect modality {dicom_file.Modality}, DICOM file with SOP instance UID {dicom_file.SOPInstanceUID} in series {series_instance_uid} rejected (row ID: {row_data['row_id']})")
                continue

            if dicom_file.StudyInstanceUID != row_data["study_instance_uid"]:
                logging.warning(f"DICOM Study Instance UID differs from the report value, DICOM file with SOP instance UID {dicom_file.SOPInstanceUID} in series {series_instance_uid} rejected (row ID: {row_data['row_id']})")
                continue

            if dicom_file.PatientID != row_data["patient_id"]:
                logging.warning(f"DICOM Patient ID differs from the report value, DICOM file with SOP instance UID {dicom_file.SOPInstanceUID} in series {series_instance_uid} rejected (row ID: {row_data['row_id']})")
                continue

            if dicom_file.PatientBirthDate != row_data["patient_birth_date"]:
                logging.warning(f"DICOM Patient Birth Date differs from the report value, DICOM file with SOP instance UID {dicom_file.SOPInstanceUID} in series {series_instance_uid} rejected (row ID: {row_data['row_id']})")
                continue

            if dicom_file.StudyDate != row_data["study_date"]:
                logging.warning(f"DICOM Study Date differs from the report value, DICOM file with SOP instance UID {dicom_file.SOPInstanceUID} in series {series_instance_uid} rejected (row ID: {row_data['row_id']})")
                continue

            # Ignore multi-frame images.
            if hasattr(dicom_file, "NumberOfFrames") and dicom_file.NumberOfFrames != 1:
                logging.warning(f"Multi-frame DICOM file with {dicom_file.NumberOfFrames} frames and SOP instance UID {dicom_file.SOPInstanceUID} in series {series_instance_uid} rejected (row ID: {row_data['row_id']})")
                continue

            # Ignore non-grayscale images.
            if dicom_file.SamplesPerPixel != 1:
                logging.warning(f"Incorrect number of samples per pixel, should be 1 but got {dicom_file.SamplesPerPixel} instead, DICOM file with SOP instance UID {dicom_file.SOPInstanceUID} in series {series_instance_uid} rejected (row ID: {row_data['row_id']})")
                continue

            # Validate histogram.
            numpy_array = dicom_utils.get_dicom_image_from_dataset(dicom_file, {"window_width": 0, "window_center": 0})
            image = Image.fromarray(numpy_array)
            res, err = image_utils.validate_image_histogram(image)
            if not res:
                logging.warning(f"Histogram validation failed: {err}, DICOM file with SOP instance UID {dicom_file.SOPInstanceUID} in series {series_instance_uid} rejected (row ID: {row_data['row_id']})")
                continue

            # Create DICOM content.
            dicom_content = "\n".join(dicom_utils.read_all_dicom_tags_from_dataset(dicom_file))
            dicom_content = (
                f"DICOM content:\n"
                f"{dicom_content}"
            )

            # Upload DICOM content file.
            dicom_content_file_name = instances_dir.replace("/", "_") + "_" + dicom_file.SOPInstanceUID + ".txt"
            dicom_content_file_name = os.path.join(config.DESTINATION_GCS_IMAGES_DIR, dicom_content_file_name)
            upload_data = [
                {"is_file": False, "local_file_or_string": dicom_content, "gcs_file_name": dicom_content_file_name},
            ]
            res, err_msg = gcs_utils.upload_files(upload_data=upload_data, gcs_bucket_name=config.DESTINATION_GCS_BUCKET_NAME)
            if not res:
                logging.warning(f"Error uploading, DICOM file with SOP instance UID {dicom_file.SOPInstanceUID} in series {series_instance_uid} rejected (row ID: {row_data['row_id']})")
                continue


def get_row_data(row):
    try:
        row_data = {
            "row_id": row["row_id"],
            "accession_number": row["AccessionNumber"],
            "study_instance_uid": row["StudyInstanceUid"],
            "series_instance_uids": ast.literal_eval(row["SeriesInstanceUid"]),
            "modalities": ast.literal_eval(row["Modality"]),
            "patient_id": row["PatientID"],
            "patient_birth_date": str(int(row["PatientBirthDate"])) if not math.isnan(row["PatientBirthDate"]) else "",
            "study_date": str(int(row["StudyDate"])) if not math.isnan(row["StudyDate"]) else "",
            "institution_name": row["InstitutionName"]
        }

        return row_data

    except Exception as e:
        return None


def get_dicom_files_from_gcs(gcs_bucket_name, gcs_dir):
    client = None

    try:
        client = storage.Client()
        bucket = client.bucket(gcs_bucket_name)
        blobs = bucket.list_blobs(prefix=gcs_dir)
        blobs = [blob for blob in blobs if blob.name.endswith(".dcm")]

        if len(blobs) == 0:
            return None, "No DICOM files"

        if config.MIN_VOLUME_DEPTH is not None and len(blobs) < config.MIN_VOLUME_DEPTH:
            return None, f"Number of DICOM files {len(blobs)} below min volume depth {config.MIN_VOLUME_DEPTH}"

        if config.MAX_VOLUME_DEPTH is not None and len(blobs) > config.MAX_VOLUME_DEPTH:
            return None, f"Number of DICOM files {len(blobs)} above max volume depth {config.MAX_VOLUME_DEPTH}"

        dicom_files = [pydicom.dcmread(BytesIO(blob.download_as_bytes())) for blob in blobs]

        return dicom_files, ""
    except Exception as e:
        return None, f"Error downloading DICOM files: {str(e)}"


def sort_dicom_files(dicom_files):
    try:
        sorted_dicom_files = sorted(dicom_files, key=lambda dicom_file: dicom_file.InstanceNumber)
        return sorted_dicom_files
    except:
        return None


def validate_dicom_files(dicom_files, row_data):
    for dicom_file in dicom_files:
        if pydicom.tag.Tag("ImageType") not in dicom_file:
            return False, f"ImageType tag not present in the DICOM file"

        if "PRIMARY" not in dicom_file.ImageType:
            return False, f"Ignoring non-primary image type {dicom_file.ImageType}"

        if "LOCALIZER" in dicom_file.ImageType:
            return False, f"Ignoring localizer image type {dicom_file.ImageType}"

        if dicom_file.SOPClassUID not in config.DICOM_MODALITIES_MAPPING:
            return False, f"Unsupported SOP Class UID {pydicom.uid.UID(dicom_file.SOPClassUID).name}"

        if config.DICOM_MODALITIES_MAPPING[dicom_file.SOPClassUID] not in config.MODALITIES:
            return False, f"Incorrect SOP Class UID {pydicom.uid.UID(dicom_file.SOPClassUID).name}"

        if dicom_file.Modality not in config.MODALITIES:
            return False, f"Incorrect modality {dicom_file.Modality}"

        if dicom_file.StudyInstanceUID != row_data["study_instance_uid"]:
            return False, f"DICOM Study Instance UID differs from the report value"

        if dicom_file.PatientID != row_data["patient_id"]:
            return False, f"DICOM Patient ID differs from the report value"

        if dicom_file.PatientBirthDate != row_data["patient_birth_date"]:
            return False, f"DICOM Patient Birth Date differs from the report value"

        if dicom_file.StudyDate != row_data["study_date"]:
            return False, f"DICOM Study Date differs from the report value"

        # Ignore non-grayscale images.
        if dicom_file.SamplesPerPixel != 1:
            return False, f"Incorrect number of samples per pixel, should be 1 but got {dicom_file.SamplesPerPixel} instead"

        # Ignore volumes with varying slice sizes.
        if dicom_file.Rows != dicom_files[0].Rows or dicom_file.Columns != dicom_files[0].Columns:
            return False, f"Varying slice sizes detected (first slice: {dicom_files[0].Columns}x{dicom_files[0].Rows}, " \
                          f"current slice: {dicom_file.Columns}x{dicom_file.Rows})"

    # Make sure the slices are in ascending and incrementing order.
    if not all(dicom_files[i].InstanceNumber + 1 == dicom_files[i + 1].InstanceNumber for i in range(len(dicom_files) - 1)):
        return False, f"Slices are not in ascending and incrementing order"

    return True, ""


def create_nifti_volume(dicom_files):
    try:
        with warnings.catch_warnings(record=True) as w:
            images = [dicom_utils.get_dicom_image_from_dataset(dicom_file, {"window_center": 0, "window_width": 0}) for dicom_file in dicom_files]
            volume = nifti_utils.numpy_images_to_nifti_volume(images)

            # Raise error on warning.
            if len(w) > 0:
                raise RuntimeError(f"Warning encountered: {w[0].message}")

            return volume, ""
    except Exception as e:
        return None, f"Error creating NIfTI volume: {str(e)}"


def upload_nifti_volume(volume, volume_info, nifti_file_name):
    try:
        sitk.WriteImage(volume, nifti_file_name, useCompression=True)
        gcs_nifti_file_name = os.path.join(config.DESTINATION_GCS_IMAGES_DIR, nifti_file_name)
        gcs_volume_info_file_name = gcs_nifti_file_name[:-7] + ".txt"  # Minus '.nii.gz' plus '.txt'.
        upload_data = [
            {"is_file": True, "local_file_or_string": nifti_file_name, "gcs_file_name": gcs_nifti_file_name},
            {"is_file": False, "local_file_or_string": volume_info, "gcs_file_name": gcs_volume_info_file_name},
        ]
        return gcs_utils.upload_files(upload_data=upload_data, gcs_bucket_name=config.DESTINATION_GCS_BUCKET_NAME)
    except Exception as e:
        return False, f"Error uploading NIfTI volume: {str(e)}"
    finally:
        if os.path.exists(nifti_file_name):
            os.remove(nifti_file_name)
