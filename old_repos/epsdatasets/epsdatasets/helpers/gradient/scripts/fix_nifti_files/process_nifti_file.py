import logging
import os
from io import BytesIO

import pydicom
from google.cloud import storage

from epsutils.dicom import dicom_utils
from epsutils.nifti import nifti_utils

import config


def process_nifti_file(nifti_file):
    try:
        # Create GCS client.
        client = storage.Client()

        # Obtain DICOM dir from the NIfTI file name.
        dicoms_dir = os.path.join(config.GRADIENT_GCS_IMAGES_DIR, nifti_file.replace(".nii.gz", "").replace("_", "/"))

        # Get DICOM files.
        bucket = client.bucket(config.GRADIENT_GCS_BUCKET_NAME)
        blobs = bucket.list_blobs(prefix=dicoms_dir)
        blobs = [blob for blob in blobs if blob.name.endswith(".dcm")]
        if len(blobs) == 0:
            logging.error(f"No DICOM files found in {dicoms_dir}")
            return

        # Download DICOM files.
        dicom_datasets = [pydicom.dcmread(BytesIO(blob.download_as_bytes())) for blob in blobs]
        dicom_datasets = sorted(dicom_datasets, key=lambda dicom_dataset: dicom_dataset.InstanceNumber)

        # Convert DICOM files to NIfTI.
        try:
            nifti_generator = "dicom2nifti"
            nifti_utils.dicom_datasets_to_nifti_file_advanced(dicom_datasets=dicom_datasets, output_nifti_file_name=nifti_file)
        except Exception as e:
            nifti_generator = "epsutils"
            logging.warning(f"Fail-safe conversion due to: {str(e)} ({dicoms_dir})")
            nifti_utils.dicom_datasets_to_nifti_file_basic(dicom_datasets=dicom_datasets, output_nifti_file_name=nifti_file)

        # Create volume info file.
        dicom_content = "\n".join(dicom_utils.read_all_dicom_tags_from_dataset(dicom_datasets[0]))
        volume_info = (
            f"NIfTI generator: {nifti_generator}\n"
            f"Number of slices: {len(dicom_datasets)}\n"  # TODO: Get actual number of slices from NIfTI file and compare with number of DICOM files.
            f"DICOM content:\n"
            f"{dicom_content}"
        )

        # Upload NIfTI file.
        bucket = client.bucket(config.EPSILON_DESTINATION_GCS_BUCKET_NAME)
        gcs_nifti_file = os.path.join(config.EPSILON_DESTINATION_GCS_IMAGES_DIR, nifti_file)
        blob = bucket.blob(gcs_nifti_file)
        blob.upload_from_filename(nifti_file)

        # Upload volume info file.
        gcs_volume_info_file = os.path.join(config.EPSILON_DESTINATION_GCS_IMAGES_DIR, nifti_file.replace(".nii.gz", ".txt"))
        blob = bucket.blob(gcs_volume_info_file)
        blob.upload_from_string(volume_info)

    except Exception as e:
        logging.error(f"{str(e)} ({dicoms_dir})")
    finally:
        del client
        if os.path.exists(nifti_file):
            os.remove(nifti_file)
