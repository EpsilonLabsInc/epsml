import argparse
import logging
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial

from tqdm import tqdm

from epsutils.dicom import dicom_utils
from epsutils.image import image_utils
from epsutils.logging import logging_utils


def save_image(dicom_file, target_image_size, target_extension, source_dir, output_dir):
    try:
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
        logging.error(f"Error saving {dicom_file}: {str(e)}")
        print(f"Error saving {dicom_file}: {str(e)}")


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
                                source_dir=args.dicom_dir,
                                output_dir=args.output_dir)

    with ProcessPoolExecutor() as executor:  # Use default number of workers.
        results = list(tqdm(executor.map(custom_save_image, [dicom_file for dicom_file in dicom_files]), total=len(dicom_files), desc="Saving images"))


if __name__ == "__main__":
    DICOM_DIR = "/mnt/sfs-segmed-1"
    OUTPUT_DIR = "/mnt/png/512x512/segmed/batch1"
    TARGET_IMAGE_SIZE = (512, 512)
    TARGET_EXTENSION = "png"

    args = argparse.Namespace(dicom_dir=DICOM_DIR,
                              output_dir=OUTPUT_DIR,
                              target_image_size=TARGET_IMAGE_SIZE,
                              target_extension=TARGET_EXTENSION)

    main(args)
