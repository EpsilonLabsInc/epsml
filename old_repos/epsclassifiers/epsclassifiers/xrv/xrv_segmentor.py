from enum import Enum
from typing import List

import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
from PIL import Image

from epsutils.dicom import dicom_utils
from epsutils.image import image_utils


class BodyPart(Enum):
    LUNGS = 0
    HEART = 1


class XrvSegmentor:
    def __init__(self):
        self.__model = xrv.baseline_models.chestx_det.PSPNet()
        self.__model.eval()
        self.__transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(512)])

    def segment(self, image, body_parts: List[BodyPart]):
        converted_image = self.__convert_image(image)
        torch_image, pil_image = self.__preprocess_image(converted_image)

        with torch.no_grad():
            output = self.__model(torch_image)

        prob = torch.sigmoid(output)
        pred = (prob > 0.5).int().numpy()

        masks = []
        for body_part in body_parts:
            if body_part == BodyPart.LUNGS:
                masks.append(np.logical_or(pred[0, 4], pred[0, 5]))
            elif body_part == BodyPart.HEART:
                masks.append(pred[0, 8])
            else:
                raise ValueError(f"Unsupported body part {body_part}")

        return {"image": pil_image, "segmentation_masks": masks}

    def __convert_image(self, image):
        if isinstance(image, str) and image.endswith(".dcm"):
            return self.__dicom_to_image(image)
        elif isinstance(image, str):
            return Image.open(image)
        elif isinstance(image, Image.Image):
            return image
        else:
            raise ValueError("Unsupported input image")

    def __preprocess_image(self, image):
        output_image = np.array(image)
        output_image = xrv.datasets.normalize(output_image, 255)  # Convert 8-bit image to [-1024, 1024] range.
        output_image = output_image.mean(2)[None, ...]  # Make single color channel.
        output_image = self.__transform(output_image)

        image = output_image[0]
        image = (image - image.min()) / (image.max() - image.min()) * 255
        pil_image = Image.fromarray(image.astype(np.uint8))
        torch_image = torch.from_numpy(output_image)

        return torch_image, pil_image

    def __dicom_to_image(self, dicom_file):
        image = dicom_utils.get_dicom_image(dicom_file, custom_windowing_parameters={"window_center": 0, "window_width": 0})
        image = image_utils.numpy_array_to_pil_image(image, convert_to_uint8=True, convert_to_rgb=True)
        return image


if __name__ == "__main__":
    print("Running segmentation example")

    image_path = "./images/cardiomegaly_1_front.dcm"

    segmentor = XrvSegmentor()
    res = segmentor.segment(image=image_path, body_parts=[BodyPart.LUNGS, BodyPart.HEART])

    image = res["image"]
    image.save("image.png")

    for index, mask in enumerate(res["segmentation_masks"]):
        mask = (mask * 255).astype(np.uint8)
        mask = Image.fromarray(mask)
        mask.save(f"mask_{index}.png")
