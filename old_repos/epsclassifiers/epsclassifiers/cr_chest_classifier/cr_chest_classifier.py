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
from PIL import Image

from epsutils.dicom import dicom_utils
from epsutils.gcs import gcs_utils


class Label(Enum):
    NON_CHEST = 0
    CHEST = 1


class CrChestClassifier:
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

    def predict(self, images: List[Union[pydicom.dataset.FileDataset, PIL.Image.Image]], device: str) -> List[Label]:
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
            image = image.astype(np.float32)
            image = (image - image.min()) / (image.max() - image.min())
            image = Image.fromarray(image)
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
    classifier = CrChestClassifier()
    classifier.load_state_dict(torch.load("./models/cr_chest_classifier_trained_on_600k_gradient_samples.pt"))

    # The first file is non-chest, the second one is chest.
    gcs_file_names = [
        "GRADIENT-DATABASE/CR/16AG02924/GRDN0003S8F5QJ9E/GRDN6OLTOR6ULAYG/studies/1.2.826.0.1.3680043.8.498.94642734191304297204127380569653880948/series/1.2.826.0.1.3680043.8.498.10665175060707054356115909864733486966/instances/1.2.826.0.1.3680043.8.498.54248805135540844129515464186902696450.dcm",
        "GRADIENT-DATABASE/CR/22JUL2024/GRDN00EZXO2ZRJC9/GRDN4FDA9F2SEFZT/studies/1.2.826.0.1.3680043.8.498.76643895298192861155575693022906977409/series/1.2.826.0.1.3680043.8.498.99445004401831511234439921842308321863/instances/1.2.826.0.1.3680043.8.498.22847497891868892152280889192110956908.dcm"
    ]

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
