from enum import Enum
from io import BytesIO
from typing import List, Union

import cv2
import numpy as np
import PIL
import pydicom
import torch
import torchxrayvision as xrv
import torchvision.transforms as transforms

from epsutils.dicom import dicom_utils
from epsutils.gcs import gcs_utils


class Label(Enum):
    NON_FRACTURE = 0
    FRACTURE = 1


class CrFractureClassifier:
    def __init__(self):
        self.__model = xrv.models.DenseNet(num_classes=1)

        self.__transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def get_model(self):
        return self.__model

    def load_state_dict(self, state_dict):
        self.__model.load_state_dict(state_dict)

    def predict(self, images: List[Union[pydicom.dataset.FileDataset, PIL.Image.Image, np.ndarray]], device: str) -> List[Label]:
        """
        Accepts a list of pydicom datasets, PIL images or numpy arrays and runs inference on them.

        Args:
            images (List[Union[pydicom.dataset.FileDataset, PIL.Image.Image, np.ndarray]]): List of pydicom datasets, PIL images or numpy arrays.
            device (str): CUDA device. Can be 'cpu' or 'cuda'.

        Returns:
            List[Label]: A list of predicted labels.
        """

        # Convert images to torch tensors.
        torch_images = []
        for image in images:
            if isinstance(image, pydicom.dataset.FileDataset):
                image = dicom_utils.get_dicom_image_from_dataset(image, custom_windowing_parameters={"window_center": 0, "window_width": 0})
            elif isinstance(image, Image.Image):
                image = np.array(image)
            elif isinstance(image, np.ndarray):
                pass
            else:
                raise ValueError("Unsupported image type")

            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
            image = image_utils.numpy_array_to_pil_image(image, convert_to_uint8=False, convert_to_rgb=False)
            image = self.__transform(image)
            torch_images.append(image)

        # Stack images.
        inputs = torch.stack(torch_images)

        # Run inference.
        self.__model = self.__model.to(device)
        self.__model.eval()
        with torch.no_grad():
            outputs = self.__model(inputs.to(device))
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()

        # Get labels.
        labels = [Label(prediction.item()) for prediction in predictions]

        return labels


if __name__ == "__main__":
    print("Running prediction example")

    # Load model.
    print("Loading the model")
    classifier = CrFractureClassifier()
    classifier.load_state_dict(torch.load("./models/..."))  # TODO...

    # The first file is with fracture, the second one is without.
    gcs_file_names = []  # TODO...

    for gcs_file_name in gcs_file_names:
        # Download DICOM file.
        print("Downloading DICOM file")
        content = gcs_utils.download_file_as_bytes(gcs_bucket_name="epsilon-data-us-central1",
                                                   gcs_file_name=gcs_file_name)

        # Read DICOM file.
        print("Reading DICOM file")
        dataset = pydicom.dcmread(BytesIO(content))

        # Predict.
        print("Predicting")
        labels = classifier.predict(images=[dataset], device="cuda")
        print(f"Predicted label: {labels[0]}")
