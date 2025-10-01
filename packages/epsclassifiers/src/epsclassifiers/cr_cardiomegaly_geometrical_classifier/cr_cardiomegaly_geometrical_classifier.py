import numpy as np
from PIL import Image

from epsclassifiers.xrv.xrv_segmentor import XrvSegmentor, BodyPart


class CrCardiomegalyGeometricalClassifier:
    def __init__(self):
        self.__segmentor = XrvSegmentor()

    def classify(self, image):
        res = self.__segmentor.segment(image=image, body_parts=[BodyPart.LUNGS, BodyPart.HEART])

        image = res["image"]
        lungs_mask = res["segmentation_masks"][0]
        heart_mask = res["segmentation_masks"][1]
        classification_result = {
            "heart_to_chest_ratio": None,
            "chest_min_x": None,
            "chest_max_x": None,
            "chest_diameter": None,
            "heart_min_x": None,
            "heart_max_x": None,
            "heart_diameter": None,
            "image": image,
            "lungs_mask": lungs_mask,
            "heart_mask": heart_mask}

        # Compute chest diameter.
        non_negative_indices = np.nonzero(lungs_mask > 0)
        x_coordinates = non_negative_indices[1]

        if x_coordinates.size == 0:
            return classification_result

        chest_min_x = np.min(x_coordinates)
        chest_max_x = np.max(x_coordinates)
        chest_diameter = chest_max_x - chest_min_x

        if chest_diameter == 0:
            return classification_result

        # Compute heart diameter.
        non_negative_indices = np.nonzero(heart_mask > 0)
        x_coordinates = non_negative_indices[1]

        if x_coordinates.size == 0:
            return classification_result

        heart_min_x = np.min(x_coordinates)
        heart_max_x = np.max(x_coordinates)
        heart_diameter = heart_max_x - heart_min_x

        # Compute heart-to-chest ratio.
        ratio = heart_diameter / chest_diameter

        classification_result["heart_to_chest_ratio"] = ratio
        classification_result["chest_min_x"] = chest_min_x
        classification_result["chest_max_x"] = chest_max_x
        classification_result["chest_diameter"] = chest_diameter
        classification_result["heart_min_x"] = heart_min_x
        classification_result["heart_max_x"] = heart_max_x
        classification_result["heart_diameter"] = heart_diameter

        return classification_result


if __name__ == "__main__":
    print("Running classification example")

    image_path = "./images/cardiomegaly_1_front.dcm"

    classifier = CrCardiomegalyGeometricalClassifier()
    res = classifier.classify(image=image_path)

    image = res["image"]
    image.save("image.png")

    lungs_mask = res["lungs_mask"]
    lungs_mask = (lungs_mask * 255).astype(np.uint8)
    lungs_mask = Image.fromarray(lungs_mask)
    lungs_mask.save(f"lungs_mask.png")

    heart_mask = res["heart_mask"]
    heart_mask = (heart_mask * 255).astype(np.uint8)
    heart_mask = Image.fromarray(heart_mask)
    heart_mask.save(f"heart_mask.png")

    ratio = res["heart_to_chest_ratio"]
    print(f"Heart-to-chest ratio = {ratio}")
