import concurrent.futures
import os
import random
import shutil
import tempfile
import time

import numpy as np
import SimpleITK as sitk
from google.cloud import storage
from PIL import Image
from tqdm import tqdm

from epsutils.dicom import dicom_utils


def structured_dicom_files_to_nifti_files(structured_dicom_files, base_dir, output_dir, gcs_bucket_name=None, max_workers=20, perform_sanity_check=False):
    """
    Converts structured DICOM files to NIfTI volumes and either saves them to the disk or uploads them to the GCS bucket.

    Parameters:
    structured_dicom_files (list of dict): Structured DICOM files organized as:
                                           [
                                               {"path": "path_to_dicom_files_1", "dicom_files": ["dicom_file_1", "dicom_file_2", ...]},
                                               {"path": "path_to_dicom_files_2", "dicom_files": ["dicom_file_1", "dicom_file_2", ...]},
                                               {"path": "path_to_dicom_files_3", "dicom_files": ["dicom_file_1", "dicom_file_2", ...]},
                                               ...
                                               {"path": "path_to_dicom_files_n", "dicom_files": ["dicom_file_1", "dicom_file_2", ...]},
                                           ]
    base_dir (string): Base directory to be prepended to the path in the structured DICOM files. If it is None, prepending is not performed.
    output_dir (string): Either local output directory (if gcs_bucket_name is None) or output directory in the GCS bucket (if gcs_bucket_name is not None).
    gcs_bucket_name (string): Name of the GCS bucket to upload NIfTI files to.
    max_workers (int): Maximum number of concurrent workers.
    perform_sanity_check (bool): If True, performs sanity check by selecting random volumes and copying both corresponding DICOM files and NIfTI volume to
                                 a separate location for a human validator to check later.

    Returns:
    None
    """

    if gcs_bucket_name is None:
        os.makedirs(output_dir, exist_ok=True)

    num_files = len(structured_dicom_files)
    sanity_check_paths = []

    if perform_sanity_check:
        # Pick one hundred random volumes for the purpose of the sanity check.
        indexes = list(range(num_files))
        random_indexes = random.sample(indexes, 100)
        print(f"Selected random volumes with the following indexes: {random_indexes}")
        sanity_check_paths = [structured_dicom_files[index]["path"] for index in random_indexes]

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for item in structured_dicom_files:
            futures.append(executor.submit(__structured_dicom_files_to_nifti_files_worker, item, base_dir, output_dir, gcs_bucket_name, sanity_check_paths))

        for future in tqdm(concurrent.futures.as_completed(futures), total=num_files, desc="Processing"):
            future.result()


def __structured_dicom_files_to_nifti_files_worker(item, base_dir, output_dir, gcs_bucket_name=None, sanity_check_paths=None):
    images = None
    client = None
    nii_file_path = None
    dicom_info_file_path = None

    try:
        path = item["path"]
        dicom_files = item["dicom_files"]

        if len(dicom_files) == 0:
            return

        nii_file = path.replace('/', '_') + ".nii.gz"
        dicom_info_file = path.replace('/', '_') + ".txt"

        if gcs_bucket_name is not None:
            client = storage.Client()
            bucket = client.bucket(gcs_bucket_name)
            nii_blob = bucket.blob(os.path.join(output_dir, nii_file))
            info_blob = bucket.blob(os.path.join(output_dir, dicom_info_file))
            if nii_blob.exists() and info_blob.exists():
                return

        full_path = os.path.join(base_dir, path) if base_dir is not None else path
        images = [dicom_utils.get_dicom_image(os.path.join(full_path, dicom_file), {"window_center": 0, "window_width": 0}) for dicom_file in dicom_files]
        volume = numpy_images_to_nifti_volume(images)

        dicom_content = dicom_utils.read_all_dicom_tags(os.path.join(full_path, dicom_files[0]))
        dicom_content = "\n".join(dicom_content)

        is_sanity_check_sample = path in sanity_check_paths
        sanity_check_sample_index = sanity_check_paths.index(path) if is_sanity_check_sample else -1

        if gcs_bucket_name is None:
            nii_file_path = os.path.join(output_dir, nii_file)
            sitk.WriteImage(volume, nii_file_path, useCompression=True)

            dicom_info_file_path = os.path.join(output_dir, dicom_info_file)
            with open(dicom_info_file_path, "w") as file:
                file.write(dicom_content)

            if is_sanity_check_sample:
                # Copy NIfTI file.
                sanity_dir = os.path.join(output_dir, "sanity_check", str(sanity_check_sample_index + 1))
                os.makedirs(sanity_dir, exist_ok=True)
                shutil.copy(nii_file_path, os.path.join(sanity_dir, nii_file))
                # Copy DICOM files.
                [shutil.copy(os.path.join(full_path, dicom_file), os.path.join(sanity_dir, dicom_file)) for dicom_file in dicom_files]
        else:
            # TODO:
            # SimpleITK does not support in-memory I/O operations, therefore, the file needs to be stored locally
            # before being uploaded to the GCS bucket. On the other hand, NiBabel package supports in-memory I/O
            # operations but has orientation issues.
            nii_file_path = os.path.join(tempfile.gettempdir(), nii_file)
            sitk.WriteImage(volume, nii_file_path, useCompression=True)
            blob = bucket.blob(os.path.join(output_dir, nii_file))  # Destination name of the file.
            blob.upload_from_filename(nii_file_path)  # Local file to be uploaded.

            dicom_info_file_path = os.path.join(tempfile.gettempdir(), dicom_info_file)
            with open(dicom_info_file_path, "w") as file:
                file.write(dicom_content)
            blob = bucket.blob(os.path.join(output_dir, dicom_info_file))  # Destination name of the file.
            blob.upload_from_filename(dicom_info_file_path)  # Local file to be uploaded.

            if is_sanity_check_sample:
                # Copy NIfTI file.
                sanity_dir = os.path.join(output_dir, "sanity_check", str(sanity_check_sample_index + 1))
                blob = bucket.blob(os.path.join(sanity_dir, nii_file))  # Destination name of the file.
                blob.upload_from_filename(nii_file_path)  # Local file to be uploaded.
                # Copy DICOM files.
                [bucket.blob(os.path.join(sanity_dir, dicom_file)).upload_from_filename(
                    os.path.join(full_path, dicom_file)) for dicom_file in dicom_files]
    except Exception as e:
        print(f"Error: {e} --> for DICOM files in '{path}'")
    finally:
        if images:
            del images
        if client:
            del client
        if gcs_bucket_name is not None and nii_file_path and os.path.exists(nii_file_path):
            os.remove(nii_file_path)
        if gcs_bucket_name is not None and dicom_info_file_path and os.path.exists(dicom_info_file_path):
            os.remove(dicom_info_file_path)


def numpy_images_to_nifti_volume(images):
    """
    Converts a list of numpy images to a NIfTI volume.

    Parameters:
    images (list of numpy.ndarray): List of 2D numpy images.

    Returns:
    SimpleITK.Image: A SimpleITK volume created from the input images.
    """

    nifti_images = [sitk.GetImageFromArray(image) for image in images]
    volume = sitk.JoinSeries(nifti_images)

    return volume


def nifti_file_to_pil_images(nifti_file, source_data_type, target_data_type, target_image_size=None, normalization_depth=None, sample_slices=False):
    sitk_image = sitk.ReadImage(nifti_file)
    return nifti_volume_to_pil_images(sitk_image=sitk_image,
                                      source_data_type=source_data_type,
                                      target_data_type=target_data_type,
                                      target_image_size=target_image_size,
                                      normalization_depth=normalization_depth,
                                      sample_slices=sample_slices)


def nifti_volume_to_pil_images(sitk_image, source_data_type, target_data_type, target_image_size=None, normalization_depth=None, sample_slices=False):
    # Get numpy array.
    sitk_image_array = sitk.GetArrayFromImage(sitk_image)

    # Normalize image.
    if normalization_depth is not None and sitk_image_array.shape[0] != normalization_depth:
        if sample_slices:
            if normalization_depth > sitk_image_array.shape[0]:
                padding = (normalization_depth - sitk_image_array.shape[0]) // 2
                indices = list(range(sitk_image_array.shape[0]))
                sitk_image_array = sitk_image_array[indices, :, :]
                padded_array = np.zeros((normalization_depth, sitk_image_array.shape[1], sitk_image_array.shape[2]), dtype=sitk_image_array.dtype)
                padded_array[padding:padding+len(indices)] = sitk_image_array
                sitk_image_array = padded_array
            else:
                indices = np.random.choice(sitk_image_array.shape[0], normalization_depth, replace=False)
                indices = np.sort(indices)
                sitk_image_array = sitk_image_array[indices, :, :]
        else:
            new_shape = (normalization_depth, sitk_image_array.shape[1], sitk_image_array.shape[2])
            sitk_image_array = math_utils.interpolate_volume(input_volume=sitk_image_array, new_shape=new_shape)

    # Split image into a list of slices.
    slices = [sitk_image_array[i, :, :] for i in range(sitk_image_array.shape[0])]

    # Make sure all the slices are in proper format.
    assert all(slice.dtype == source_data_type for slice in slices)

    # Convert slices to PIL images.
    if target_data_type == np.float32:
        slices = [Image.fromarray(slice.astype(np.float32), mode="F") for slice in slices]
    else:
        raise NotImplementedError(f"Function not implemented yet for {target_data_type} data type")

    # Resize if needed.
    if target_image_size is not None:
        slices = [slice.resize((target_image_size, target_image_size)) for slice in slices]

    return slices
