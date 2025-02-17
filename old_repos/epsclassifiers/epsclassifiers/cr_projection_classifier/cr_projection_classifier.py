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
from epsutils.image import image_utils


class Label(Enum):
    FRONTAL_PROJECTION = 0
    LATERAL_PROJECTION = 1
    OTHER_PROJECTION = 2


class CrProjectionClassifier:
    def __init__(self):
        self.__model = xrv.models.DenseNet(num_classes=2)

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
            images (List[Union[pydicom.dataset.FileDataset, PIL.Image.Image]]): List of pydicom datasets, PIL images or numpy arrays.
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

        # Get labels.
        max_probs, max_indices = torch.max(probabilities, dim=1)
        labels = [Label(max_indices[i].item()) if max_probs[i].item() >= 0.5 else Label.OTHER_PROJECTION for i in range(len(max_probs))]

        return labels


if __name__ == "__main__":
    print("Running prediction example")

    # Load model.
    print("Loading the model")
    classifier = CrProjectionClassifier()
    classifier.load_state_dict(torch.load("./models/cr_projection_classifier_trained_on_500k_gradient_samples.pt"))

    # The first file represents a frontal view and the second one a lateral.
    gcs_file_names = [
        "gs://epsilon-data-us-central1/GRADIENT-DATABASE/CR/22JUL2024/GRDNFX1NEHGCZ3NO/GRDNMZ596YJR9CJB/studies/1.2.826.0.1.3680043.8.498.85293159377516994281827729047946673763/series/1.2.826.0.1.3680043.8.498.31034184508339455314298357221886973531/instances/1.2.826.0.1.3680043.8.498.99703763766863412540749061652026413739.dcm",
        "gs://epsilon-data-us-central1/GRADIENT-DATABASE/CR/22JUL2024/GRDNFHIBVLFP33JN/GRDNZQSUV7SOKU9O/studies/1.2.826.0.1.3680043.8.498.47406028988612654352842040157436800865/series/1.2.826.0.1.3680043.8.498.96807073672981820339274799922843052349/instances/1.2.826.0.1.3680043.8.498.64897527864475258723018780022660824170.dcm"
    ]

    for gcs_file_name in gcs_file_names:
        # Download DICOM file.
        print("Downloading DICOM file")
        gcs_data = gcs_utils.split_gcs_uri(gcs_file_name)
        content = gcs_utils.download_file_as_bytes(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

        # Read DICOM file.
        print("Reading DICOM file")
        dataset = pydicom.dcmread(BytesIO(content))

        # Predict.
        print("Predicting...")
        labels = classifier.predict(images=[dataset], device="cuda")
        print(f"Predicted label: {labels[0]}")
