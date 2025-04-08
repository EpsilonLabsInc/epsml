import ast
import os
from typing import Callable, Optional

import pandas as pd

from epsutils.gcs import gcs_utils
from epsutils.dicom import dicom_utils
from epsutils.image import image_utils

from .extended import ExtendedVisionDataset


class GradientDataset(ExtendedVisionDataset):
    def __init__(
        self,
        images_file,
        dir_prefix_to_remove,
        dir_prefix_to_add,
        remove_deid_from_path,
        root = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform)

        self.__images_file = images_file
        self.__dir_prefix_to_remove = dir_prefix_to_remove
        self.__dir_prefix_to_add = dir_prefix_to_add
        self.__remove_deid_from_path = remove_deid_from_path in ["True", "true"] if isinstance(remove_deid_from_path, str) else remove_deid_from_path

        print("----------------------------------")
        print("GradientDataset is using the following configuration:")
        print(f"- Images file: {self.__images_file}")
        print(f"- Dir prefix to remove: {self.__dir_prefix_to_remove}")
        print(f"- Dir prefix to add: {self.__dir_prefix_to_add}")
        print(f"- Remove deid from path: {self.__remove_deid_from_path}")
        print("----------------------------------")

        self.__generate_dataset()

    def __generate_dataset(self):
        # Download or load images file.
        if gcs_utils.is_gcs_uri(self.__images_file):
            print("Downloading the images file")
            gcs_data = gcs_utils.split_gcs_uri(self.__images_file)
            content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])
        else:
            print("Loading images file")
            with open(self.__images_file, "r") as file:
                content = file.read()

        # Populate the data.
        print("Populating the data")
        data = []
        rows = content.splitlines()
        for index, row in enumerate(rows):
            row = ast.literal_eval(row)
            image_path = row["image_path"]
            image_path = os.path.relpath(image_path, self.__dir_prefix_to_remove) if self.__dir_prefix_to_remove else image_path
            if self.__remove_deid_from_path:
                image_path = image_path.replace("/deid/", "/")
            data.append({"image_path": os.path.join(self.__dir_prefix_to_add, image_path), "labels": row["labels"]})

        # Create dataset.
        print("Creating the dataset")
        self.__pandas_dataset = pd.DataFrame(data)

        # Print dataset.
        print(f"The dataset has {len(self.__pandas_dataset)} rows")
        print("Dataset head:")
        print(self.__pandas_dataset.head())

    def get_image_data(self, index):
        try:
            item = self.__pandas_dataset.iloc[index]
            image_path = item["image_path"]

            image = dicom_utils.get_dicom_image(image_path, custom_windowing_parameters={"window_center": 0, "window_width": 0})
            image = image_utils.numpy_array_to_pil_image(image, convert_to_uint8=True, convert_to_rgb=True)
            data = image_utils.pil_image_to_byte_stream(image)
            return data
        except Exception as e:
            print(f"Error loading {item['image_path']}: {str(e)}")
            raise

    def get_target(self, index):
        return 0  # Return dummy label since it's not needed for self supervised learning.

    def __len__(self):
        return len(self.__pandas_dataset)
