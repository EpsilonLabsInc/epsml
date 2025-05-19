import ast

from tqdm import tqdm

from epsutils.gcs import gcs_utils
from epsutils.image import image_augmentation


def get_images(images_file):
    print("Downloading images file")

    gcs_data = gcs_utils.split_gcs_uri(images_file)
    content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

    print("Generating a list of images")

    images = []
    rows = content.splitlines()

    for row in rows:
        image = ast.literal_eval(row)
        images.append(image)

    return images


def get_labels(images):
    print(f"Generating a list of labels for {len(images)} images")

    labels = []

    for image in tqdm(images, total=len(images), desc="Processing"):
        labels.append(image["labels"])

    return labels


def get_labels_distribution(labels):
    print("Calculating labels distribution")

    composite_labels_distribution = {}
    single_labels_distribution = {}

    for label in tqdm(labels, total=len(labels), desc="Processing"):
        if label == []:
            print(f"WARNING: Empty label")
            continue

        if len(label) == 1:
            if label[0] in single_labels_distribution:
                single_labels_distribution[label[0]] += 1
            else:
                single_labels_distribution[label[0]] = 1

        for l in label:
            if l in composite_labels_distribution:
                composite_labels_distribution[l] += 1
            else:
                composite_labels_distribution[l] = 1

    composite_labels_distribution = dict(sorted(composite_labels_distribution.items(), key=lambda item: item[1], reverse=True))
    single_labels_distribution = dict(sorted(single_labels_distribution.items(), key=lambda item: item[1], reverse=True))

    return {"composite_labels_distribution": composite_labels_distribution, "single_labels_distribution": single_labels_distribution}


def augment_labels(labels, labels_to_augment):
    augmented_labels = labels.copy()
    augmentation_factors = []

    for label in labels:
        max_augmentation_factor = 0

        for label_name, augmentation_factor in labels_to_augment.items():
            if label_name in label:
                max_augmentation_factor = max(augmentation_factor, max_augmentation_factor)

        augmentation_factors.append(max_augmentation_factor)

        for i in range(max_augmentation_factor):
            augmented_labels.append(label)

    return augmented_labels, augmentation_factors


def split_no_findings(images, no_findings_split_factor):
    print(f"Splitting 'No Findings' labels by split factor of {no_findings_split_factor}")

    num_no_findings = 0

    for image in images:
        if image["labels"] == ["No Findings"]:
            num_no_findings += 1

    num_no_findings_to_keep = num_no_findings // no_findings_split_factor

    num_no_findings = 0
    filtered_images = []

    for image in images:
        if image["labels"] != ["No Findings"]:
            filtered_images.append(image)
            continue

        if num_no_findings < num_no_findings_to_keep:
            filtered_images.append(image)

        num_no_findings += 1

    print(f"Num images before/after splitting: {len(images)}/{len(filtered_images)}")

    return filtered_images


def augment_images(images, augmentation_factors, seed):
    assert len(images) == len(augmentation_factors)

    # Compute number of required augmentations.
    num_required_augmentations = 0
    for augmentation_factor in augmentation_factors:
        num_required_augmentations += augmentation_factor

    # Generate random augmentation parameters.
    augmentation_parameters = image_augmentation.generate_augmentation_parameters(num_images=num_required_augmentations,
                                                                                  min_rotation_in_degrees=-5,
                                                                                  max_rotation_in_degrees=+5,
                                                                                  seed=seed)

    augmented_images = images.copy()

    # Original images do not need any augmentation parameters.
    for augmented_image in augmented_images:
        augmented_image["augmentation_parameters"] = None

    augmentation_index = 0

    # Start augmenting.
    for image, augmentation_factor in zip(images, augmentation_factors):
        for i in range(augmentation_factor):
            augmented_image = image.copy()
            augmented_image["augmentation_parameters"] = augmentation_parameters[augmentation_index]
            augmented_images.append(augmented_image)
            augmentation_index += 1

    assert len(augmentation_parameters) == augmentation_index

    return augmented_images
