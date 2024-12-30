from typing import Optional

import numpy as np


VALIDATE_IMAGE_HISTOGRAM_CONFIGURATIONS = {
    "CHEST_CR_SCAN": {
        "lateral_entropy_min": 5.5,
        "lateral_entropy_max": 8.0,
        "ap_entropy_min": 5.25,
        "ap_entropy_max": 7.75
    },
    "NON_CHEST_CR_SCAN": {
        "lateral_entropy_min": 5.25,
        "lateral_entropy_max": 8.0,
        "ap_entropy_min": 5.25,
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
            if entropy < lateral_entropy_min or entropy > lateral_entropy_max:
                return False, f"Entropy {entropy:.2f} invalid for lateral view"
        else:  # AP view.
            if entropy < ap_entropy_min or entropy > ap_entropy_max:
                return False, f"Entropy {entropy:.2f} invalid for AP view"

        return True, ""

    except Exception as e:
        return False, f"{e}"
