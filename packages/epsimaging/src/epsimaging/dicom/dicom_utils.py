import re
from dateutil.parser import parse
from enum import Enum
from typing import Tuple, List

import numpy as np
import pydicom
import SimpleITK as sitk

from epsutils.dicom import dicom_compression_utils


def get_friendly_names(sop_class_uids: Tuple[pydicom.uid.UID, ...]):
    return tuple([sop_class_uid.name for sop_class_uid in sop_class_uids])


def get_friendly_tag_name(tag_str: str):
    str_list = tag_str.strip('()').split(', ')
    hex_list = [int(f"0x{s}", 16) for s in str_list]
    tag = pydicom.tag.Tag(hex_list)
    return pydicom.datadict.keyword_for_tag(tag)


def compare_strings_without_carets(str1, str2):
    filtered_str1 = " ".join(str1.replace("^", " ").strip().split())
    filtered_str2 = " ".join(str2.replace("^", " ").strip().split())
    return filtered_str1 == filtered_str2


def normalize_name(name):
    # Replace non-alpha characters with space, convert to uppercase,
    # remove leading & trailing spaces, split into parts and sort them.
    name_list = re.sub(r"[^A-Za-z]", " ", name).upper().strip().split()
    name_list.sort()
    normalized_name = ', '.join(name_list)
    return normalized_name


def compare_names(name1, name2):
    return normalize_name(name1) == normalize_name(name2)


def compare_dates(date_str1, date_str2, allow_empty_dates=True):
    if allow_empty_dates:
        if date_str1.strip() == date_str2.strip():
            return True

    date1 = parse(date_str1).strftime("%Y%m%d")
    date2 = parse(date_str2).strftime("%Y%m%d")
    return date1 == date2


def age_to_years(age):
    age = str(age)

    if age.upper().endswith("D"):
        return 0
    elif age.upper().endswith("M"):
        return 0
    elif age.upper().endswith("Y"):
        return int(age[:-1])
    else:
        return int(age)


def read_dicom_tags_from_dataset(dataset: pydicom.dataset.FileDataset, tags):
    values = {}

    for tag in tags:
        name = tag["name"]
        required = tag["required"]

        try:
            values[name] = dataset.pixel_array if name == "PixelArray" else dataset[name].value
        except Exception as e:
            if not required:
                values[name] = None
            else:
                if type(e) is KeyError:
                    # KeyError exception is not descriptive enough, so we catch it and raise a more descriptive one.
                    tag_str = str(e)
                    raise Exception(f"DICOM tag '{get_friendly_tag_name(tag_str)}' {tag_str} not found")
                else:
                    raise

    return values


def read_dicom_tags(dicom_file_name, tags):
    dataset = pydicom.dcmread(dicom_file_name)
    return read_dicom_tags_from_dataset(dataset, tags)


def read_all_dicom_tags_from_dataset(dataset: pydicom.dataset.FileDataset, include_pixel_data=False):
    dicom_content = []

    for element in dataset:
        if element.name == "Pixel Data" and not include_pixel_data:
            continue

        dicom_content.append(f"{element.tag} {element.name}: {element.value}")

    return dicom_content


def read_all_dicom_tags(dicom_file_name, include_pixel_data=False):
    dataset = pydicom.dcmread(dicom_file_name)
    return read_all_dicom_tags_from_dataset(dataset, include_pixel_data)


def generate_dicom_dict_from_dataset(dataset: pydicom.dataset.FileDataset, include_pixel_data=False):
    dicom_dict = {}

    for element in dataset:
        if element.name == "Pixel Data" and not include_pixel_data:
            continue

        dicom_dict[f"{element.tag} {element.name}"] = str(element.value)

    return dicom_dict


def generate_dicom_dict(dicom_file_name, include_pixel_data=False):
    dataset = pydicom.dcmread(dicom_file_name)
    return generate_dicom_dict_from_dataset(dataset, include_pixel_data)


def get_dicom_image_from_dataset(dataset: pydicom.dataset.FileDataset, custom_windowing_parameters=None):
    pixel_array = dataset.pixel_array

    # Handle PhotometricInterpretation.
    if dataset.PhotometricInterpretation == "MONOCHROME1":
        pixel_array = np.max(pixel_array) - pixel_array
    elif dataset.PhotometricInterpretation == "MONOCHROME2":
        pass
    else:
        raise ValueError(f"Unsupported PhotometricInterpretation '{dataset.PhotometricInterpretation}'")

    # Handle rescaling.
    pixel_array = pydicom.pixels.apply_modality_lut(pixel_array, dataset)

    # Apply windowing operation.
    if custom_windowing_parameters is None:
        # Use windowing parameters from DICOM file.
        if "VOILUTSequence" in dataset or ("WindowCenter" in dataset and "WindowWidth" in dataset):
            pixel_array = pydicom.pixels.apply_voi_lut(pixel_array, dataset, index=0)
    else:
        if custom_windowing_parameters["window_center"] == 0 and custom_windowing_parameters["window_width"] == 0:
            # Skip windowing operation.
            pass
        else:
            # Use custom windowing parameters.
            min_val = custom_windowing_parameters["window_center"] - (custom_windowing_parameters["window_width"] / 2)
            max_val = custom_windowing_parameters["window_center"] + (custom_windowing_parameters["window_width"] / 2)
            pixel_array = np.clip(pixel_array, min_val, max_val)

    return pixel_array


def get_dicom_image(dicom_file_name, custom_windowing_parameters=None):
    dataset = pydicom.dcmread(dicom_file_name)
    return get_dicom_image_from_dataset(dataset, custom_windowing_parameters)


def get_dicom_image_fail_safe(dicom_file_name, custom_windowing_parameters=None):
    try:
        dataset = pydicom.dcmread(dicom_file_name)
    except:
        dataset = pydicom.dcmread(dicom_file_name, force=True)
        dataset = dicom_compression_utils.handle_dicom_compression(dataset)

    return get_dicom_image_from_dataset(dataset, custom_windowing_parameters)


def get_dicom_image_fast(dicom_file_name):
    return sitk.GetArrayFromImage(sitk.ReadImage(dicom_file_name)).squeeze()  # Remove batch dimension using squeeze().


def check_dicom_image_in_dataset(dataset: pydicom.dataset.FileDataset):
    return "PixelData" in dataset


def check_dicom_image(dicom_file_name):
    dataset = pydicom.dcmread(dicom_file_name)
    return check_dicom_image_in_dataset(dataset)


def check_dicom_volume_images_quality(volume):
    """
    Checks the quality of the DICOM volume images.

    Parameters:
    volume (list):
        Volume as a list of slices, where each slice is a dictionary of corresponding DICOM values and a numpy image.
        For example:
        volume = [
            {"dicom_values": {"WindowCenter": 1, "WindowWidth": 1}, "image": <image_1>},
            {"dicom_values": {"WindowCenter": 2, "WindowWidth": 2}, "image": <image_2>},
            {"dicom_values": {"WindowCenter": 3, "WindowWidth": 3}, "image": <image_3>},
            ...
            {"dicom_values": {"WindowCenter": 4, "WindowWidth": 4}, "image": <image_4>}]

    Returns:
    tuple: A tuple containing:
        - res (bool): True if image quality check is successful, False otherwise.
        - err_msg (str): An empty string if image quality check is successful, otherwise an error message describing why image quality check failed.
    """

    # Check std dev of window centers.
    std_dev = np.std([slice["dicom_values"]["WindowCenter"] for slice in volume])
    if std_dev != 0.0:
        return False, f"Std dev of window centers {std_dev} != 0.0"

    # Check std dev of window widths.
    std_dev = np.std([slice["dicom_values"]["WindowWidth"] for slice in volume])
    if std_dev != 0.0:
        return False, f"Std dev of window widths {std_dev} != 0.0"

    return True, ""


class AnatomicalPlane(Enum):
    UNKNOWN = -1
    AXIAL = 0
    CORONAL = 1
    SAGITTAL = 2


def get_anatomical_plane(image_orientation: List[float]) -> AnatomicalPlane:
    if len(image_orientation) != 6:
        raise ValueErrur("Image orientation should be vector of length 6")

    row = image_orientation[:3]
    max_index = np.argmax(np.abs(row))
    result = [0, 0, 0]
    result[max_index] = 1 if row[max_index] >= 0 else -1
    row = result

    col = image_orientation[3:]
    max_index = np.argmax(np.abs(col))
    result = [0, 0, 0]
    result[max_index] = 1 if col[max_index] >= 0 else -1
    col = result

    if row == [1, 0, 0] and col == [0, 1, 0]:
        return AnatomicalPlane.AXIAL
    elif row == [1, 0, 0] and col == [0, 0, -1]:
        return AnatomicalPlane.CORONAL
    elif row == [0, 1, 0] and col == [0, 0, -1]:
        return AnatomicalPlane.SAGITTAL
    else:
        return AnatomicalPlane.UNKNOWN
