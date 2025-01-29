import ast
import os
from io import BytesIO

import datasets
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from epsdatasets.helpers.base.base_dataset_helper import BaseDatasetHelper
from epsutils.dicom import dicom_utils
from epsutils.gcs import gcs_utils
from epsutils.labels import labels_utils
from epsutils.labels.cr_chest_labels import EXTENDED_CR_CHEST_LABELS


class GradientCrDatasetHelper(BaseDatasetHelper):
    def __init__(self, gcs_train_file, gcs_validation_file, gcs_test_file=None, images_dir=None, dir_prefix_to_remove=None, convert_images_to_rgb=True, custom_labels=None):
        """
        Initializes the GradientCrDatasetHelper with the specified parameters.

        Args:
            gcs_train_file (str): GCS URI of the training file.
            gcs_validation_file (str): GCS URI of the validation file.
            gcs_test_file (str, optional): GCS URI of the test file.
            images_dir (str, optional): GCS URI or local path of the base folder where the images are located.
            dir_prefix_to_remove (str, optional): Dir prefix to be removed from the input image paths.
            convert_images_to_rgb (bool, optional): If True, loaded images will be converted to RGB before being returned.
            custom_labels ((List[str], optional): Custom labels to be used. If None, default EXTENDED_CR_CHEST_LABELS will be used.
        """
        super().__init__(gcs_train_file=gcs_train_file,
                         gcs_validation_file=gcs_validation_file,
                         gcs_test_file=gcs_test_file,
                         images_dir=images_dir,
                         dir_prefix_to_remove=dir_prefix_to_remove,
                         convert_images_to_rgb=convert_images_to_rgb,
                         custom_labels=custom_labels)

    def _load_dataset(self, *args, **kwargs):
        # Store params.
        self.__gcs_train_file = kwargs["gcs_train_file"] if "gcs_train_file" in kwargs else next((arg for arg in args if arg == "gcs_train_file"), None)
        self.__gcs_validation_file = kwargs["gcs_validation_file"] if "gcs_validation_file" in kwargs else next((arg for arg in args if arg == "gcs_validation_file"), None)
        self.__gcs_test_file = kwargs["gcs_test_file"] if "gcs_test_file" in kwargs else next((arg for arg in args if arg == "gcs_test_file"), None)
        self.__images_dir = kwargs["images_dir"] if "images_dir" in kwargs else next((arg for arg in args if arg == "images_dir"), None)
        self.__dir_prefix_to_remove = kwargs["dir_prefix_to_remove"] if "dir_prefix_to_remove" in kwargs else next((arg for arg in args if arg == "dir_prefix_to_remove"), None)
        self.__convert_images_to_rgb = kwargs["convert_images_to_rgb"] if "convert_images_to_rgb" in kwargs else next((arg for arg in args if arg == "convert_images_to_rgb"), None)
        self.__custom_labels = kwargs["custom_labels"] if "custom_labels" in kwargs else next((arg for arg in args if arg == "custom_labels"), None)

        self.__pandas_full_dataset = None
        self.__pandas_train_dataset = None
        self.__pandas_validation_dataset = None
        self.__pandas_test_dataset = None
        self.__torch_train_dataset = None
        self.__torch_validation_dataset = None
        self.__torch_test_dataset = None

        # Download training file.
        print("Downloading the training file")
        gcs_data = gcs_utils.split_gcs_uri(self.__gcs_train_file)
        content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

        # Populate training data.
        print("Populating the training data")
        data = []
        rows = content.splitlines()
        for row in rows:
            row = ast.literal_eval(row)
            image_path = row["image_path"]
            image_path = os.path.relpath(image_path, self.__dir_prefix_to_remove) if self.__dir_prefix_to_remove else image_path
            data.append({"image_path": os.path.join(self.__images_dir, image_path), "labels": row["labels"]})

        # Create traning dataset.
        print("Creating the training dataset")
        self.__pandas_train_dataset = pd.DataFrame(data)

        # Print training dataset.
        print(f"The training dataset has {len(self.__pandas_train_dataset)} rows")
        print("Training dataset head:")
        print(self.__pandas_train_dataset.head())

        # Download validation file.
        print("Downloading the validation file")
        gcs_data = gcs_utils.split_gcs_uri(self.__gcs_validation_file)
        content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

        # Populate validation data.
        print("Populating the validation data")
        data = []
        rows = content.splitlines()
        for row in rows:
            row = ast.literal_eval(row)
            image_path = row["image_path"]
            image_path = os.path.relpath(image_path, self.__dir_prefix_to_remove) if self.__dir_prefix_to_remove else image_path
            data.append({"image_path": os.path.join(self.__images_dir, image_path), "labels": row["labels"]})

        # Create validation dataset.
        print("Creating the validation dataset")
        self.__pandas_validation_dataset = pd.DataFrame(data)

        # Print validation dataset.
        print(f"The validation dataset has {len(self.__pandas_validation_dataset)} rows")
        print("Validation dataset head:")
        print(self.__pandas_validation_dataset.head())

        if self.__gcs_test_file is not None:
            # Download test file.
            print("Downloading the test file")
            gcs_data = gcs_utils.split_gcs_uri(self.__gcs_test_file)
            content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

            # Populate test data.
            print("Populating the test data")
            data = []
            rows = content.splitlines()
            for row in rows:
                row = ast.literal_eval(row)
                image_path = row["image_path"]
                image_path = os.path.relpath(image_path, self.__dir_prefix_to_remove) if self.__dir_prefix_to_remove else image_path
                data.append({"image_path": os.path.join(self.__images_dir, image_path), "labels": row["labels"]})

            # Create test dataset.
            print("Creating the test dataset")
            self.__pandas_test_dataset = pd.DataFrame(data)

            # Print test dataset.
            print(f"The test dataset has {len(self.__pandas_test_dataset)} rows")
            print("Test dataset head:")
            print(self.__pandas_test_dataset.head())

        # Create Torch datasets.
        print("Creating Torch datasets")
        self.__torch_train_dataset = GradientCrTorchDataset(pandas_dataframe=self.__pandas_train_dataset)
        self.__torch_validation_dataset = GradientCrTorchDataset(pandas_dataframe=self.__pandas_validation_dataset)
        self.__torch_test_dataset = GradientCrTorchDataset(pandas_dataframe=self.__pandas_test_dataset) if self.__pandas_test_dataset else None

    def get_pil_image(self, item):
        try:
            image_path = item["image_path"]

            if gcs_utils.is_gcs_uri(image_path):
                gcs_data = gcs_utils.split_gcs_uri(image_path)
                image_path = BytesIO(gcs_utils.download_file_as_bytes(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"]))

            image = dicom_utils.get_dicom_image(image_path, custom_windowing_parameters={"window_center": 0, "window_width": 0})
            image = image.astype(np.float32)
            eps = 1e-10
            image = (image - image.min()) / (image.max() - image.min() + eps)

            if self.__convert_images_to_rgb:
                image = image * 255
                image = image.astype(np.uint8)

            image = Image.fromarray(image)

            if self.__convert_images_to_rgb:
                image = image.convert("RGB")

            return image

        except Exception as e:
            print(f"Error loading {item['image_path']}: {str(e)}")
            raise

    def get_torch_image(self, item, processor):
        raise NotImplementedError("Not implemented")

    def get_labels(self):
        if self.__custom_labels:
            return self.__custom_labels
        else:
            return EXTENDED_CR_CHEST_LABELS

    def get_ids_to_labels(self):
        raise NotImplementedError("Not implemented")

    def get_labels_to_ids(self):
        raise NotImplementedError("Not implemented")

    def get_torch_label(self, item):
        return torch.tensor(labels_utils.to_multi_hot_encoding(item["labels"], self.__custom_labels if self.__custom_labels else EXTENDED_CR_CHEST_LABELS))

    def get_pandas_full_dataset(self):
        raise NotImplementedError("Not implemented")

    def get_hugging_face_train_dataset(self):
        return datasets.Dataset.from_pandas(self.__pandas_train_dataset)

    def get_hugging_face_validation_dataset(self):
        return datasets.Dataset.from_pandas(self.__pandas_validation_dataset)

    def get_hugging_face_test_dataset(self):
        return datasets.Dataset.from_pandas(self.__pandas_test_dataset) if self.__pandas_test_dataset else None

    def get_torch_train_dataset(self):
         return self.__torch_train_dataset

    def get_torch_validation_dataset(self):
        return self.__torch_validation_dataset

    def get_torch_test_dataset(self):
        return self.__torch_test_dataset

    def get_torch_train_data_loader(self, collate_function, batch_size, num_workers):
        data_loader = DataLoader(self.__torch_train_dataset,
                                 collate_fn=collate_function,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 persistent_workers=True)
        return data_loader

    def get_torch_validation_data_loader(self, collate_function, batch_size, num_workers):
        data_loader = DataLoader(self.__torch_validation_dataset,
                                 collate_fn=collate_function,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 persistent_workers=True)
        return data_loader

    def get_torch_test_data_loader(self, collate_function, batch_size, num_workers):
        data_loader = DataLoader(self.__torch_test_dataset,
                                 collate_fn=collate_function,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 persistent_workers=True) if self.__torch_test_dataset else None
        return data_loader


class GradientCrTorchDataset(Dataset):
    def __init__(self, pandas_dataframe):
        self.__pandas_dataframe = pandas_dataframe

    def __getitem__(self, idx):
        item = self.__pandas_dataframe.iloc[idx]
        return item

    def __len__(self):
        return len(self.__pandas_dataframe)
