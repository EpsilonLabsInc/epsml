from enum import Enum
from typing import List, Union

import PIL
import pydicom
import torch
import torchxrayvision as xrv
import torchvision.transforms as transforms
from PIL import Image

from epsutils.dicom import dicom_utils


class Label(Enum):
    NON_CHEST = 0
    CHEST = 1


class CrChestClassifier:
    def __init__(self):
        self.__model = xrv.models.DenseNet(num_classes=1)

        self.__transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def get_model(self):
        return self.__model

    def load_state_dict(self, state_dict):
        self.__model.load_state_dict(state_dict)

    def predict(self, images: List[Union[pydicom.dataset.FileDataset, PIL.Image.Image]], device: str) -> List[Label]:
        """
        Accepts a list of pydicom datasets or PIL images and runs inference on them.

        Args:
            images (List[Union[pydicom.dataset.FileDataset, PIL.Image.Image]]): List of pydicom datasets or PIL images.
            device (str): CUDA device. Can be 'cpu' or 'cuda'.

        Returns:
            List[Label]: A list of predicted labels.
        """

        # Convert images to torch tensors.
        torch_images = []
        for image in images:
            if isinstance(image, pydicom.dataset.FileDataset):
                image = dicom_utils.get_dicom_image(image, custom_windowing_parameters={"window_center": 0, "window_width": 0})
                image = image.astype(np.float32)
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                image = Image.fromarray(image)
            elif isinstance(image, Image.Image):
                pass
            else:
                raise ValueError("Unsupported image type")

            image = self.__transform(image)
            torch_images.append(image)

        # Stack images.
        inputs = torch.stack(torch_images)

        # Run inference.
        self.__model.eval()
        with self.__model.no_grad():
            outputs = self.__model(inputs.to(device))
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()

        # Get labels.
        labels = [Label(prediction) for prediction in predictions]

        return labels
