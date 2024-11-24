import os
from io import BytesIO

import pydicom
import SimpleITK as sitk
from google.cloud import storage

from epsutils.dicom import dicom_utils
from epsutils.nifti import nifti_utils

GRADIENT_GCS_BUCKET_NAME = "epsilon-data-us-central1"
IMAGES_DIR = "GRADIENT-DATABASE/CT/01OCT2024/"
REFERENCE_NIFTI_FILE_NAME = "GRDN0007QSTCWBGT_GRDNR0EUF06IQS7I_studies_1.2.826.0.1.3680043.8.498.25233327885027416478309187654544542912_series_1.2.826.0.1.3680043.8.498.13818303120429167839490207877085371300_instances.nii.gz"
USE_WINDOW_CENTER_AND_WIDTH_FROM_DICOM = False
SAVE_DICOMS = True
OUTPUT_NIFTI_FILE_NAME = "c:/users/andrej/desktop/output.nii.gz"


def main():
    dicom_datasets = []

    if GRADIENT_GCS_BUCKET_NAME is None:
        for filename in os.listdir(IMAGES_DIR):
            dicom_datasets.append(pydicom.dcmread(os.path.join(IMAGES_DIR, filename)))
    else:
        client = storage.Client()
        bucket = client.bucket(GRADIENT_GCS_BUCKET_NAME)
        prefix = IMAGES_DIR if REFERENCE_NIFTI_FILE_NAME is None else os.path.join(IMAGES_DIR, REFERENCE_NIFTI_FILE_NAME.replace(".nii.gz", "").replace("_", "/"))
        blobs = bucket.list_blobs(prefix=prefix)
        blobs = [blob for blob in blobs if blob.name.endswith(".dcm")]

        if SAVE_DICOMS:
            local_dir = prefix.replace("/", "_")
            os.makedirs(local_dir, exist_ok=True)

            for blob in blobs:
                local_file = os.path.join(local_dir, os.path.basename(blob.name))
                blob.download_to_filename(local_file)
                dicom_datasets.append(pydicom.dcmread(local_file))
        else:
            dicom_datasets = [pydicom.dcmread(BytesIO(blob.download_as_bytes())) for blob in blobs]

    # Sort DICOM files by instance number.
    dicom_datasets = sorted(dicom_datasets, key=lambda dicom_dataset: dicom_dataset.InstanceNumber)

    # Get images from DICOM files.
    custom_windowing_parameters = None if USE_WINDOW_CENTER_AND_WIDTH_FROM_DICOM else {"window_center": 0, "window_width": 0}
    images = [dicom_utils.get_dicom_image_from_dataset(dataset, custom_windowing_parameters=custom_windowing_parameters) for dataset in dicom_datasets]

    # Create NIfTI volume.
    volume = nifti_utils.numpy_images_to_nifti_volume(images)

    # Save to NIfTI file.
    sitk.WriteImage(volume, OUTPUT_NIFTI_FILE_NAME, useCompression=True)


if __name__ == "__main__":
    main()
