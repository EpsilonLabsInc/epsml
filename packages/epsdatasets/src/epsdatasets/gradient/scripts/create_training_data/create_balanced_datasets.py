import argparse
import json
from enum import Enum

from sklearn.model_selection import train_test_split

from epsutils.labels.cr_chest_labels import CR_CHEST_LABELS
from epsutils.misc import misc_utils
from epsutils.visualization import visualization_utils

import helpers


class ProgramMode(Enum):
    STATISTICS = 1
    MANUAL_AUGMENTATION_SIMULATION = 2
    AUTOMATIC_AUGMENTATION_SIMULATION = 3
    DATASET_GENERATION = 4


def run_statistics(images_file):
    images = helpers.get_images(images_file)
    labels = helpers.get_labels(images)
    distribution = helpers.get_labels_distribution(labels)

    print("Composite labels distribution:")
    print(distribution["composite_labels_distribution"])

    fig, _ = visualization_utils.generate_histogram(data=distribution["composite_labels_distribution"],
                                                    x_labels_rotation_angle=90,
                                                    show_x_labels_with_values=True)

    fig.savefig("composite_labels_distribution.png")

    print("Single labels distribution:")
    print(distribution["single_labels_distribution"])

    fig, _ = visualization_utils.generate_histogram(data=distribution["single_labels_distribution"],
                                                    x_labels_rotation_angle=90,
                                                    show_x_labels_with_values=True)

    fig.savefig("single_labels_distribution.png")


def run_manual_augmentation_simulation(images_file):
    images = helpers.get_images(images_file)
    labels = helpers.get_labels(images)

    while True:
        print("")
        print("---------------------------------------------------------------------------------------")
        print("Enter the IDs and augmentation factors of the labels you want to augment or 'q' to quit")
        print(f"Label IDs are: {misc_utils.list_to_indexed_dict(CR_CHEST_LABELS)}")
        print("---------------------------------------------------------------------------------------")
        print("")

        inp = input("")

        if inp == "q":
            break

        try:
            labels_to_augment = dict(item.split(":") for item in inp.split(","))
            labels_to_augment = {CR_CHEST_LABELS[int(label_id.strip())]: int(augmentation_factor.strip()) for label_id, augmentation_factor in labels_to_augment.items()}
        except:
            print("Incorrect command format")
            continue

        print(f"Labels to augment {labels_to_augment}")

        augmented_labels, _ = helpers.augment_labels(labels=labels, labels_to_augment=labels_to_augment)
        distribution = helpers.get_labels_distribution(augmented_labels)

        print("Composite labels distribution after augmentation:")
        print(distribution["composite_labels_distribution"])


def run_automatic_augmentation_simulation(images_file):
    raise ValueError("Not implemented yet")


def run_dataset_generation(args):
    # Get images.
    images = helpers.get_images(args.images_file)

    # Remove excess No Findings labels.
    images = helpers.split_no_findings(images, args.no_findings_split_factor)

    # Calculate labels distribution before augmentation.
    labels = helpers.get_labels(images)
    distribution = helpers.get_labels_distribution(labels)

    print("Composite labels distribution before augmentation:")
    print(distribution["composite_labels_distribution"])

    # Augment images.
    _, augmentation_factors = helpers.augment_labels(labels=labels, labels_to_augment=args.labels_to_augment)
    images = helpers.augment_images(images=images, augmentation_factors=augmentation_factors, seed=args.seed)

    # Calculate labels distribution after augmentation.
    labels = helpers.get_labels(images)
    distribution = helpers.get_labels_distribution(labels)

    print("Composite labels distribution after augmentation:")
    print(distribution["composite_labels_distribution"])

    # Create splits.
    train_set, temp_set = train_test_split(images, test_size=0.10, random_state=args.seed)
    val_set, test_set = train_test_split(temp_set, test_size=0.50, random_state=args.seed)

    print(f"Training dataset size: {len(train_set)}")
    labels = helpers.get_labels(train_set)
    distribution = helpers.get_labels_distribution(labels)
    print("Composite labels distribution of training dataset:")
    print(distribution["composite_labels_distribution"])

    print(f"Validation dataset size: {len(val_set)}")
    labels = helpers.get_labels(val_set)
    distribution = helpers.get_labels_distribution(labels)
    print("Composite labels distribution of validation dataset:")
    print(distribution["composite_labels_distribution"])

    print(f"Test dataset size: {len(test_set)}")
    labels = helpers.get_labels(test_set)
    distribution = helpers.get_labels_distribution(labels)
    print("Composite labels distribution of test dataset:")
    print(distribution["composite_labels_distribution"])

    print("Saving datasets")

    with open(args.output_training_file, "w") as file:
        for item in train_set:
            json.dump(item, file)
            file.write("\n")

    with open(args.output_validation_file, "w") as file:
        for item in val_set:
            json.dump(item, file)
            file.write("\n")

    with open(args.output_test_file, "w") as file:
        for item in test_set:
            json.dump(item, file)
            file.write("\n")


def main(args):
    if args.program_mode == ProgramMode.STATISTICS:
        run_statistics(args.images_file)

    elif args.program_mode == ProgramMode.MANUAL_AUGMENTATION_SIMULATION:
        run_manual_augmentation_simulation(args.images_file)

    elif args.program_mode == ProgramMode.AUTOMATIC_AUGMENTATION_SIMULATION:
        run_automatic_augmentation_simulation(args.images_file)

    elif args.program_mode == ProgramMode.DATASET_GENERATION:
        run_dataset_generation(args)

    else:
        raise ValueError("Not implemented")


if __name__ == "__main__":
    IMAGES_FILE = "gs://gradient-crs/archive/training/self-supervised-training/gradient-crs-all-batches-chest-images-448x448-png-with-labels.json"
    PROGRAM_MODE = ProgramMode.DATASET_GENERATION
    NO_FINDINGS_SPLIT_FACTOR = 2
    LABELS_TO_AUGMENT = {"Enlarged Cardiomediastinum": 19, "Pneumothorax": 2, "Lung Lesion": 2, "Pleural Other": 1, "Consolidation": 1}
    SEED = 42
    OUTPUT_TRAINING_FILE = "gradient-crs-all-batches-chest-images-448x448-png-with-labels-train.json"
    OUTPUT_VALIDATION_FILE = "gradient-crs-all-batches-chest-images-448x448-png-with-labels-validation.json"
    OUTPUT_TEST_FILE = "gradient-crs-all-batches-chest-images-448x448-png-with-labels-test.json"

    args = argparse.Namespace(images_file=IMAGES_FILE,
                              program_mode=PROGRAM_MODE,
                              no_findings_split_factor=NO_FINDINGS_SPLIT_FACTOR,
                              labels_to_augment=LABELS_TO_AUGMENT,
                              seed=SEED,
                              output_training_file=OUTPUT_TRAINING_FILE,
                              output_validation_file=OUTPUT_VALIDATION_FILE,
                              output_test_file=OUTPUT_TEST_FILE)

    main(args)
