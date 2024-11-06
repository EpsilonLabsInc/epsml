import argparse
import ast
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from epsutils.dicom import dicom_utils


def mean_std_dataset(args):
    print(f"Loading generated data from '{args.generated_data_file}'")
    df = pd.read_csv(args.generated_data_file)
    df = df.map(ast.literal_eval)  # Make sure all the elements are converted from strings back to original Python types.

    volumes = df["volume"].to_dict().values()
    num_volumes = len(volumes)
    custom_windowing_parameters = {"window_center": 0, "window_width": 0}

    mean = 0.0
    std = 0.0
    num_samples = 0
    progress_bar = tqdm(volumes, total=num_volumes, desc="Scanning volumes")

    for volume in progress_bar:
        path = os.path.join(args.images_dir, volume["path"])
        dicom_files = [os.path.join(path, dicom_file) for dicom_file in volume["dicom_files"]]
        num_dicom_files = len(dicom_files)

        if num_dicom_files < 3:
            continue

        middle_index = num_dicom_files // 2
        indices = [middle_index - 1, middle_index, middle_index + 1]

        for i in indices:
            image = dicom_utils.get_dicom_image(dicom_file_name=dicom_files[i], custom_windowing_parameters=custom_windowing_parameters)
            assert image.dtype == np.uint16
            mean += np.mean(image)
            std += np.std(image)
            num_samples += 1

        if progress_bar.n > 0 and progress_bar.n % 20000 == 0:
            temp_mean = mean / num_samples
            temp_std = std / num_samples
            temp_norm_mean = temp_mean / 65535
            temp_norm_std = temp_std / 65535
            print(f"Intermediate mean/std: {temp_mean:.4f}/{temp_std:.4f}, "
                  f"intermediate normalized mean/std: {temp_norm_mean:.4f}/{temp_norm_std:.4f}")

        time.sleep(0.01)

    mean /= num_samples
    std /= num_samples
    print("Mean:", mean)
    print("Standard deviation:", std)

    # Normalize to range [0, 1].
    mean /= 65535
    std /= 65535
    print("Normalized mean:", mean)
    print("Normalized standard deviation:", std)

def mean_std_image(args):
    custom_windowing_parameters = {"window_center": 0, "window_width": 0}
    image = dicom_utils.get_dicom_image(dicom_file_name=args.dicom_file, custom_windowing_parameters=custom_windowing_parameters)

    mean = np.mean(image)
    std = np.std(image)
    print("Mean:", mean)
    print("Standard deviation:", std)

    # Normalize to range [0, 1].
    mean /= 65535
    std /= 65535
    print("Normalized mean:", mean)
    print("Normalized standard deviation:", std)


def main():
    parser = argparse.ArgumentParser(description="Gradient dataset analysis")
    parser.add_argument("--analysis_type", choices=["mean_std_dataset", "mean_std_image"], required=True, help="Type of analysis.")

    args, unknown = parser.parse_known_args()

    if args.analysis_type == "mean_std_dataset":
        parser.add_argument("--generated_data_file", required=True, help="Generated data file.")
        parser.add_argument("--images_dir", required=True, help="Images directory.")
    elif args.analysis_type == "mean_std_image":
        parser.add_argument("--dicom_file", required=True, help="DICOM file.")
    else:
        raise ValueError(f"Unsupported analysis type '{analysis_type}'")

    args = parser.parse_args()

    if args.analysis_type == "mean_std_dataset":
        mean_std_dataset(args)
    elif args.analysis_type == "mean_std_image":
        mean_std_image(args)
    else:
        raise ValueError(f"Unsupported analysis type '{analysis_type}'")


if __name__ == "__main__":
    main()
