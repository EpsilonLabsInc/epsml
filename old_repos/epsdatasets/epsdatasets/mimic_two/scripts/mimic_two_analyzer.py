import argparse
import os
from collections import defaultdict

import pandas as pd
from PIL import Image

IMAGE_FILENAMES_FILE = "IMAGE_FILENAMES"
IMAGE_FILENAMES_CORRECTED_FILE = "IMAGE_FILENAMES_CORRECTED"
METADATA_FILE = "mimic-cxr-2.0.0-metadata.csv.gz"
SPLIT_FILE = "mimic-cxr-2.0.0-split.csv.gz"
CHEXPERT_FILE = "mimic-cxr-2.0.0-chexpert.csv.gz"
NEGBIO_FILE = "mimic-cxr-2.0.0-negbio.csv.gz"


def count_images_in_directory(directory):
    count = 0
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            if os.path.splitext(filename)[1].lower() in [".jpg"]:
                count += 1
    return count


def find_directory(path, directory_name):
    for root, dirs, _ in os.walk(path):
        if directory_name in dirs:
            return os.path.join(root, directory_name)
    return None


def run_analysis(directory):
    total_images = 0
    images_per_study = defaultdict(list)
    max_images_per_study = 0

    for dir_path, _, _ in os.walk(directory):
        print(f"\rScanning directory '{dir_path}'", end="")
        image_count = count_images_in_directory(dir_path)
        total_images += image_count
        images_per_study[image_count].append(dir_path)

        if image_count > max_images_per_study:
            max_images_per_study = image_count

    print("... done")

    return total_images, images_per_study, max_images_per_study


def main():
    parser = argparse.ArgumentParser(description="Mimic Two dataset analysis")
    parser.add_argument("dataset_path", type=str, help="Dataset path")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--analyze", help="Run analysis", action="store_true")
    group.add_argument("--preview_study", type=int, metavar="STUDY_ID", help="Preview study images")
    group.add_argument("--unique_positions", help="Get all unique positions in the dataset", action="store_true")
    args = parser.parse_args()

    if args.analyze:
        total_images, images_per_study, max_images_per_study = run_analysis(args.dataset_path)
        print(f"Total images: {total_images}")
        print(f"Max images per study:")
        for num_images in range(max(max_images_per_study - 3, 0), max_images_per_study + 1):
            print(f"- {num_images} images in {len(images_per_study[num_images])} studies (showing only some of them):")
            for study_dir in images_per_study[num_images][:10]:
                print(f"  - '{os.path.relpath(study_dir, args.dataset_path)}'")
    elif args.preview_study:
        print(f"Looking for study '{args.preview_study}'")
        study_dir = find_directory(args.dataset_path, f"s{args.preview_study}")
        if study_dir is None:
            print("Study not found")
            exit(0)

        print(f"Study directory '{os.path.relpath(study_dir, args.dataset_path)}'")
        image_files = [os.path.join(study_dir, f) for f in os.listdir(study_dir) if f.endswith(".jpg")]
        if not image_files:
            print("No image files found for this study")
            exit(0)

        print("Found the following study images:")
        for image_file in image_files:
            print(os.path.relpath(image_file, args.dataset_path))
            image = Image.open(image_file).convert("RGB")
            image.show()
    elif args.unique_positions:
        print("Scanning dataset")
        metadata = pd.read_csv(os.path.join(args.dataset_path, METADATA_FILE), compression="gzip")
        unique_positions = metadata["ViewPosition"].unique()
        print(f"Unique positions: {unique_positions}")

if __name__ == "__main__":
    main()
