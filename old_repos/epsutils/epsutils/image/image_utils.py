from typing import Optional

import numpy as np
from PIL import Image


VALIDATE_IMAGE_HISTOGRAM_CONFIGURATIONS = {
    "CHEST_CR_SCAN": {
        "lateral_entropy_min": 5.5,
        "lateral_entropy_max": 8.0,
        "ap_entropy_min": 5.25,
        "ap_entropy_max": 7.75
    },
    "NON_CHEST_CR_SCAN": {
        "lateral_entropy_min": 0.0,  # Don't filter out non-chest images that are too dark.
        "lateral_entropy_max": 8.0,
        "ap_entropy_min": 0.0,  # Don't filter out non-chest images that are too dark.
        "ap_entropy_max": 8.0
    }
}


def validate_image_histogram(image, config=VALIDATE_IMAGE_HISTOGRAM_CONFIGURATIONS["CHEST_CR_SCAN"]):
    lateral_entropy_min = config["lateral_entropy_min"]
    lateral_entropy_max = config["lateral_entropy_max"]
    ap_entropy_min = config["ap_entropy_min"]
    ap_entropy_max = config["ap_entropy_max"]

    try:
        pixrange = image.getextrema()[1] - image.getextrema()[0]

        if pixrange < 255.0 or pixrange > 65535.0:
            return False, f"Invalid pixel range {pixrange} (must be between 255 and 65535)"

        img_array = np.array(image)
        hist = np.histogram(img_array.flatten(), bins=256)[0]
        hist = hist[hist > 0]
        prob = hist / hist.sum()
        entropy = -np.sum(prob * np.log2(prob))

        width, height = image.size
        if height > 1.1 * width:  # Lateral view.
            if entropy < lateral_entropy_min:
                return False, f"Entropy {entropy:.2f} too small for lateral view"
            if entropy > lateral_entropy_max:
                return False, f"Entropy {entropy:.2f} too big for lateral view"
        else:  # AP view.
            if entropy < ap_entropy_min:
                return False, f"Entropy {entropy:.2f} too small for AP view"
            if entropy > ap_entropy_max:
                return False, f"Entropy {entropy:.2f} too big for AP view"

        return True, ""

    except Exception as e:
        return False, f"{e}"


def numpy_array_to_pil_image(image_array, convert_to_uint8=True, convert_to_rgb=True):
    image = image_array.astype(np.float32)
    eps = 1e-10
    image = (image - image.min()) / (image.max() - image.min() + eps)

    if convert_to_uint8 or convert_to_rgb:
        image = image * 255
        image = image.astype(np.uint8)

    image = Image.fromarray(image)

    if convert_to_rgb:
        image = image.convert("RGB")

    return image


def min_bounding_rectangle(image_mask, padding_ratio=0.0, return_relative_coordinates=True):
    image_mask = np.asarray(image_mask)

    if not np.any(image_mask):
        return None

    coords = np.argwhere(image_mask)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    padding_x = int(width * padding_ratio / 2)
    padding_y = int(height * padding_ratio / 2)

    x_min = max(0, x_min - padding_x)
    y_min = max(0, y_min - padding_y)
    x_max = min(image_mask.shape[1] - 1, x_max + padding_x)
    y_max = min(image_mask.shape[0] - 1, y_max + padding_y)

    if return_relative_coordinates:
        x_min_rel = x_min / image_mask.shape[1]
        y_min_rel = y_min / image_mask.shape[0]
        x_max_rel = x_max / image_mask.shape[1]
        y_max_rel = y_max / image_mask.shape[0]

        return x_min_rel, y_min_rel, x_max_rel, y_max_rel
    else:
        return x_min, y_min, x_max, y_max


def crop_image(image, crop_coordinates, use_relative_coordinates=False):
    if use_relative_coordinates:
        width, height = image.size
        x_min = int(crop_coordinates[0] * width)
        y_min = int(crop_coordinates[1] * height)
        x_max = int(crop_coordinates[2] * width)
        y_max = int(crop_coordinates[3] * height)

        return image.crop((x_min, y_min, x_max, y_max))
    else:
        return image.crop(crop_coordinates)
