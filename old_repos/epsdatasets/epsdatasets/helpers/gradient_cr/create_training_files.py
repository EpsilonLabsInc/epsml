import ast
import json
import os
import random
from io import StringIO

import pandas as pd

from epsutils.gcs import gcs_utils
from epsutils.labels.cr_chest_labels import EXTENDED_CR_CHEST_LABELS

GCS_INPUT_FILE = "gs://epsilonlabs-filestore/cleaned_CRs/gradient_rm_bad_dcm_1211_nolabel.jsonl"
GCS_INPUT_IMAGES_DIR = "GRADIENT-DATABASE/CR/22JUL2024/"
GCS_CHEST_IMAGES_FILE = "gs://gradient-crs/archive/training/chest_files_gradient_all_3_batches.csv"
TARGET_LABELS = ["Other"]
SEED = 42
FILL_UP_VALIDATION_LIST = False
OUTPUT_TRAINING_FILE = "gradient-crs-22JUL2024-chest-images-with-other-label-training.jsonl"
OUTPUT_VALIDATION_FILE = "gradient-crs-22JUL2024-chest-images-with-other-label-validation.jsonl"


def get_labels_distribution(images):
    labels_dist = {item: 0 for item in EXTENDED_CR_CHEST_LABELS}
    newly_added_labels = set()

    for image in images:
        for label in image["labels"]:
            if label in labels_dist:
                labels_dist[label] += 1
            else:
                newly_added_labels.add(label)
                labels_dist[label] = 1

    return labels_dist, newly_added_labels


def main():
    print("1.")
    print("Downloading chest images file")

    gcs_data = gcs_utils.split_gcs_uri(GCS_CHEST_IMAGES_FILE)
    content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

    print("")
    print("2.")
    print("Generating a list of chest images")

    df = pd.read_csv(StringIO(content), header=None, sep=';')
    chest_images = set(df[0])

    print("")
    print("3.")
    print("Downloading input file")

    gcs_data = gcs_utils.split_gcs_uri(GCS_INPUT_FILE)
    content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

    print("")
    print("4.")
    print("Generating a list of input images")

    input_images = []
    rows = content.splitlines()
    for row in rows:
        row = ast.literal_eval(row)
        labels = row["labels"]
        assert labels != []
        images = row["image"]

        for image in images:
            image_path = os.path.join(GCS_INPUT_IMAGES_DIR, image)
            input_images.append({"image_path": image_path, "labels": labels})

    print(f"Number of input images: {len(input_images)}")

    print("")
    print("5.")
    print("Removing non-chest images from the input images")

    filtered_images = []
    for input_image in input_images:
        if input_image["image_path"] not in chest_images:
            continue

        filtered_images.append(input_image)

    print(f"Number of input images after non-chest removal: {len(filtered_images)}")

    labels_dist, newly_added_labels = get_labels_distribution(filtered_images)
    print(f"Labels distribution: {labels_dist}")
    print(f"Newly added labels: {newly_added_labels}")

    print("")
    print("6a.")
    print("Fixing labels: Renaming unknown labels to 'Other'")

    for image in filtered_images:
        labels = image["labels"]
        fixed_labels = []

        for label in labels:
            if label in EXTENDED_CR_CHEST_LABELS:
                fixed_labels.append(label)
            elif "Other" not in fixed_labels:
                fixed_labels.append("Other")

        image["labels"] = fixed_labels

    labels_dist, newly_added_labels = get_labels_distribution(filtered_images)
    print(f"Labels distribution: {labels_dist}")
    print(f"Newly added labels: {newly_added_labels}")

    if TARGET_LABELS:
        print("")
        print("6b.")
        print(f"Fixing labels: Applying target labels {TARGET_LABELS}")

        for image in filtered_images:
            labels = image["labels"]
            fixed_labels = []

            for label in labels:
                if label in TARGET_LABELS:
                    fixed_labels.append(label)

            image["labels"] = fixed_labels

        labels_dist, newly_added_labels = get_labels_distribution(filtered_images)
        print(f"Labels distribution: {labels_dist}")
        print(f"Newly added labels: {newly_added_labels}")

        print("")
        print("6c.")
        print(f"Fixing labels: Selecting image subset for better labels distribution")

        images_with_non_empty_labels = [image for image in filtered_images if image["labels"]]
        images_with_empty_labels = [image for image in filtered_images if not image["labels"]]
        selected_images_with_empty_labels = images_with_empty_labels[0:len(images_with_non_empty_labels)]
        remaining_images_for_validation = images_with_empty_labels[len(images_with_non_empty_labels):]
        filtered_images = images_with_non_empty_labels + selected_images_with_empty_labels

        print(f"Subset selected: {len(images_with_non_empty_labels)} images with non-empty labels, {len(selected_images_with_empty_labels)} images with empty labels")
        print(f"Remaining images for validation: {len(remaining_images_for_validation)}")

        labels_dist, newly_added_labels = get_labels_distribution(filtered_images)
        print(f"Labels distribution: {labels_dist}")
        print(f"Newly added labels: {newly_added_labels}")

    print("")
    print("7.")
    print("Creating splits")

    random.seed(SEED)
    random.shuffle(filtered_images)
    split_index = int(0.98 * len(filtered_images))
    training_set = filtered_images[:split_index]
    validation_set = filtered_images[split_index:]

    if TARGET_LABELS and FILL_UP_VALIDATION_LIST:
        validation_set.extend(remaining_images_for_validation)

    print("")
    print(f"Training set size: {len(training_set)}")
    for i in range(5):
        print(training_set[i])
    labels_dist, newly_added_labels = get_labels_distribution(training_set)
    print(f"Labels distribution: {labels_dist}")
    print(f"Newly added labels: {newly_added_labels}")

    print("")
    print(f"Validation set size: {len(validation_set)}")
    for i in range(5):
        print(validation_set[i])
    labels_dist, newly_added_labels = get_labels_distribution(validation_set)
    print(f"Labels distribution: {labels_dist}")
    print(f"Newly added labels: {newly_added_labels}")

    print("")
    print("8.")
    print("Writing training and validation set to file")

    with open(OUTPUT_TRAINING_FILE, "w") as f:
        for item in training_set:
            f.write(json.dumps(item) + "\n")

    with open(OUTPUT_VALIDATION_FILE, "w") as f:
        for item in validation_set:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
