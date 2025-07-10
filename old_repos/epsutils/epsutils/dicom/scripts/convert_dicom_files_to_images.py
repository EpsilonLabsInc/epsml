import argparse
import logging
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial

import pydicom
from tqdm import tqdm

from epsutils.dicom import dicom_utils, dicom_compression_utils
from epsutils.image import image_utils
from epsutils.logging import logging_utils


def save_image(dicom_file, target_image_size, target_extension, allowed_dicom_tag_values, source_dir, output_dir):
    try:
        dataset = pydicom.dcmread(dicom_file, force=True)
        dataset = dicom_compression_utils.handle_dicom_compression(dataset)

        if dataset.Modality not in allowed_dicom_tag_values["modalities"]:
            logging.warning(f"Unsupported modality {dataset.Modality}: {dicom_file}")
            return

        if dataset.SOPClassUID not in allowed_dicom_tag_values["sop_class_uids"]:
            logging.warning(f"Unsupported SOPClassUID {pydicom.uid.UID(dataset.SOPClassUID).name}: {dicom_file}")
            return

        image = dicom_utils.get_dicom_image_fail_safe(dicom_file, custom_windowing_parameters={"window_center": 0, "window_width": 0})
        image = image_utils.numpy_array_to_pil_image(image, convert_to_uint8=True, convert_to_rgb=True)

        if target_image_size is not None:
            image = image.resize(target_image_size)

        image_path = os.path.join(output_dir, os.path.relpath(dicom_file, source_dir))
        image_path = image_path.replace("dcm", target_extension)

        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image.save(image_path)

        logging.info(f"Image successfully saved: {image_path}")
    except Exception as e:
        logging.error(f"Failed to convert {dicom_file}: {str(e)}")
        print(f"Failed to convert {dicom_file}: {str(e)}")


def main(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging_utils.configure_logger(logger_file_name=f"dicom_to_image_conversion_{timestamp}.log", show_logging_level=True)

    print(f"Scanning {args.dicom_dir} for the DICOM files")
    dicom_dir = pathlib.Path(args.dicom_dir)
    dicom_files = list(dicom_dir.rglob("*.dcm"))
    print(f"Found {len(dicom_files)} DICOM files")

    custom_save_image = partial(save_image,
                                target_image_size=args.target_image_size,
                                target_extension=args.target_extension,
                                allowed_dicom_tag_value=args.allowed_dicom_tag_values,
                                source_dir=args.dicom_dir,
                                output_dir=args.output_dir)

    with ProcessPoolExecutor() as executor:  # Use default number of workers.
        results = list(tqdm(executor.map(custom_save_image, [dicom_file for dicom_file in dicom_files]), total=len(dicom_files), desc="Saving images"))


if __name__ == "__main__":
    DICOM_DIR = "/mnt/sfs-simonmed"
    OUTPUT_DIR = "/mnt/png/512x512/simonmed"
    TARGET_IMAGE_SIZE = (512, 512)
    TARGET_EXTENSION = "png"
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

    args = argparse.Namespace(dicom_dir=DICOM_DIR,
                              output_dir=OUTPUT_DIR,
                              target_image_size=TARGET_IMAGE_SIZE,
                              target_extension=TARGET_EXTENSION,
                              allowed_dicom_tag_values=ALLOWED_DICOM_TAG_VALUES)

    main(args)
