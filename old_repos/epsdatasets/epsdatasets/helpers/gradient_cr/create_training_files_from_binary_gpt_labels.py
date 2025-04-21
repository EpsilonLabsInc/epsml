import ast
import json
import os
import random
from io import StringIO

import pandas as pd
from tqdm import tqdm

from epsutils.gcs import gcs_utils

LABEL_COLUMN_NAME = "alveolar_expanded_labels"
TARGET_LABELS = ["Edema"]
NO_FINDINGS_LABEL = "No Findings"
TREAT_NON_TARGET_LABELS_AS_NO_FINDINGS = True
GCS_INPUT_FILE = "gs://report_csvs/cleaned/CR/labels_for_binary_classification/GRADIENT_CR_ALL_CHEST_BATCHES_cleaned_alveolar_expanded_labels.csv"
GCS_INPUT_IMAGES_DIR = "GRADIENT-DATABASE/CR"
GCS_CHEST_IMAGES_FILE = "gs://gradient-crs/archive/training/chest/chest_files_gradient_all_3_batches.csv"
GCS_FRONTAL_PROJECTIONS_FILE = "gs://gradient-crs/archive/training/projections/gradient-crs-all-batches-chest-only-frontal-projections.csv"
GCS_LATERAL_PROJECTIONS_FILE = "gs://gradient-crs/archive/training/projections/gradient-crs-all-batches-chest-only-lateral-projections.csv"
INCLUDE_REPORT_TEXT = True
GENERATE_PER_NORMALIZED_STUDY = False
GENERATE_PER_FRONTAL_LATERAL_STUDY = True
SEED = 42
IMAGE_PATH_SUBSTR_FOR_VALIDATION_DATASET = None
SPLIT_RATIO = 0.98
OUTPUT_TRAINING_FILE = "gradient-crs-all-batches-two-image-study-chest-images-with-obvious-edema-alveolar-label-with-text-training.jsonl"
OUTPUT_VALIDATION_FILE = "gradient-crs-all-batches-two-image-study-chest-images-with-obvious-edema-alveolar-label-with-text-validation.jsonl"


def get_labels_distribution(images):
    labels_dist = {item: 0 for item in TARGET_LABELS}
    newly_added_labels = set()

    for image in images:
        for label in image["labels"]:
            if label in labels_dist:
                labels_dist[label] += 1
            else:
                newly_added_labels.add(label)
                labels_dist[label] = 1

    return labels_dist, newly_added_labels


def normalize_list(lst, num_elems):
    if len(lst) == 0:
        return []
    elif len(lst) > num_elems:
        return lst[:num_elems]
    elif len(lst) == num_elems:
        return lst
    else:
        return lst + [lst[-1]] * (num_elems - len(lst))


def main():
    # Download chest images file.

    print("Downloading chest images file")

    gcs_data = gcs_utils.split_gcs_uri(GCS_CHEST_IMAGES_FILE)
    content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

    # Generate a list of chest images.

    print("")
    print("Generating a list of chest images")

    df = pd.read_csv(StringIO(content), header=None, sep=';')
    chest_images = set(df[0])

    # Download projection files if necessary.

    if GENERATE_PER_FRONTAL_LATERAL_STUDY:
        print("")
        print("Downloading frontal projections file")

        gcs_data = gcs_utils.split_gcs_uri(GCS_FRONTAL_PROJECTIONS_FILE)
        content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

        print("")
        print("Generating a list of frontal images")

        df = pd.read_csv(StringIO(content), header=None, sep=';')
        frontal_images = set(df[0])

        print("")
        print("Downloading lateral projections file")

        gcs_data = gcs_utils.split_gcs_uri(GCS_LATERAL_PROJECTIONS_FILE)
        content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

        print("")
        print("Generating a list of lateral images")

        df = pd.read_csv(StringIO(content), header=None, sep=';')
        lateral_images = set(df[0])

    # Download input file.

    print("")
    print("Downloading input file")

    gcs_data = gcs_utils.split_gcs_uri(GCS_INPUT_FILE)
    content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

    # Generate a list of input images.

    print("")
    print("Generating a list of input images")

    input_images = []

    if GCS_INPUT_FILE.endswith(".jsonl"):
        rows = content.splitlines()
        for row in rows:
            row = ast.literal_eval(row)
            labels = row["labels"]
            assert labels != []
            images = row["image"]

            for image in images:
                image_path = os.path.join(GCS_INPUT_IMAGES_DIR, image)
                input_images.append({"image_path": image_path, "labels": labels})

    elif GCS_INPUT_FILE.endswith(".csv"):
        df = pd.read_csv(StringIO(content), low_memory=False)
        df = df[[LABEL_COLUMN_NAME, "cleaned_report_text", "image_paths", "batch_id"]]
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            try:
                labels = [label.strip() for label in row[LABEL_COLUMN_NAME].split(",")]
                report_text = row["cleaned_report_text"].replace("\n", "\\n") if INCLUDE_REPORT_TEXT else None
                image_paths = ast.literal_eval(row["image_paths"])
                batch_id = row["batch_id"]
            except Exception as e:
                print(f"{str(e)}: Error parsing {row[LABEL_COLUMN_NAME]}")
                continue

            if isinstance(image_paths, dict):
                image_paths_dict = image_paths
                image_paths = []
                for value in image_paths_dict.values():
                    image_paths.extend(value["paths"])

            if batch_id in ("20DEC2024", "09JAN2025"):
                batch_id = os.path.join(batch_id, "deid")

            image_paths = [os.path.join(GCS_INPUT_IMAGES_DIR, batch_id, image_path) for image_path in image_paths]

            if GENERATE_PER_NORMALIZED_STUDY:
                image_paths = normalize_list(image_paths, num_elems=3)  # Take max 3 images per study. If less then 3, multiplicate last image to fill up the gap.
                input_images.append({"image_path": image_paths, "report_text": report_text, "labels": labels})
            elif GENERATE_PER_FRONTAL_LATERAL_STUDY:
                if len(image_paths) != 2:
                    continue

                if not any(image_path in frontal_images for image_path in image_paths) or not any(image_path in lateral_images for image_path in image_paths):
                    continue

                sorted_image_paths = [image_path for image_path in image_paths if image_path in frontal_images] + [image_path for image_path in image_paths if image_path in lateral_images]
                input_images.append({"image_path": sorted_image_paths, "report_text": report_text, "labels": labels})
            else:
                for image_path in image_paths:
                    input_images.append({"image_path": image_path, "report_text": report_text, "labels": labels})

    else:
        raise ValueError("Input file type not supported")

    print(f"Number of input images: {len(input_images)}")

    # Remove non-chest images.

    print("")
    print("Removing non-chest images from the input images")

    filtered_images = []
    for input_image in input_images:
        if isinstance(input_image["image_path"], list):
            if not all(image_path in chest_images for image_path in input_image["image_path"]):
                continue
        else:
            if input_image["image_path"] not in chest_images:
                continue

        filtered_images.append(input_image)

    print(f"Number of input images after non-chest removal: {len(filtered_images)}")

    labels_dist, newly_added_labels = get_labels_distribution(filtered_images)
    print(f"Labels distribution: {labels_dist}")
    print(f"Newly added labels: {newly_added_labels}")

    # Filter images.

    print("")
    print("Filtering images")

    input_images = filtered_images
    filtered_images = []

    for input_image in input_images:
        labels = input_image["labels"]

        if TREAT_NON_TARGET_LABELS_AS_NO_FINDINGS:
            if any(label.startswith(target_label) and label != target_label for label in labels for target_label in TARGET_LABELS):
                continue
        else:
            if not any(label in TARGET_LABELS or label == NO_FINDINGS_LABEL for label in labels):
                continue

        input_image["labels"] = [label for label in labels if label in TARGET_LABELS]
        filtered_images.append(input_image)

    print(f"Number of input images after removing images with no target labels: {len(filtered_images)}")

    labels_dist, newly_added_labels = get_labels_distribution(filtered_images)
    print(f"Labels distribution: {labels_dist}")
    print(f"Newly added labels: {newly_added_labels}")

    if IMAGE_PATH_SUBSTR_FOR_VALIDATION_DATASET is None:
        # Select image subset for better labels distribution.

        print("")
        print(f"Selecting image subset for better labels distribution")

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

        # Create splits.

        print("")
        print("Creating splits")

        random.seed(SEED)
        random.shuffle(filtered_images)

        split_index = int(SPLIT_RATIO * len(filtered_images))
        training_set = filtered_images[:split_index]
        validation_set = filtered_images[split_index:]

    else:
        # Create training and validation dataset split using IMAGE_PATH_SUBSTR_FOR_VALIDATION_DATASET.

        print("")
        print(f"Creating training and validation dataset split using image path substring '{IMAGE_PATH_SUBSTR_FOR_VALIDATION_DATASET}'")

        training_set = []
        validation_set = []

        for filtered_image in filtered_images:
            if isinstance(filtered_image["image_path"], list):
                if all(IMAGE_PATH_SUBSTR_FOR_VALIDATION_DATASET not in image_path for image_path in filtered_image["image_path"]):
                    training_set.append(filtered_image)
                elif all(IMAGE_PATH_SUBSTR_FOR_VALIDATION_DATASET in image_path for image_path in filtered_image["image_path"]):
                    validation_set.append(filtered_image)
            else:
                if IMAGE_PATH_SUBSTR_FOR_VALIDATION_DATASET not in filtered_image["image_path"]:
                    training_set.append(filtered_image)
                else:
                    validation_set.append(filtered_image)

        random.seed(SEED)
        random.shuffle(training_set)
        random.shuffle(validation_set)

        # Select subset of the training dataset for better labels distribution.

        print("")
        print(f"Selecting subset of the training dataset for better labels distribution")

        images_with_non_empty_labels = [image for image in training_set if image["labels"]]
        images_with_empty_labels = [image for image in training_set if not image["labels"]]
        selected_images_with_empty_labels = images_with_empty_labels[0:len(images_with_non_empty_labels)]
        training_set = images_with_non_empty_labels + selected_images_with_empty_labels
        random.shuffle(training_set)

        print(f"Subset selected: {len(images_with_non_empty_labels)} images with non-empty labels, {len(selected_images_with_empty_labels)} images with empty labels")

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

    # Write output files.

    print("")
    print("Writing training and validation set to file")

    with open(OUTPUT_TRAINING_FILE, "w") as f:
        for item in training_set:
            f.write(json.dumps(item) + "\n")

    with open(OUTPUT_VALIDATION_FILE, "w") as f:
        for item in validation_set:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
