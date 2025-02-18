import json
import os
from io import BytesIO, StringIO

import cv2
import datasets
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from epsdatasets.helpers.base.base_dataset_helper import BaseDatasetHelper
from epsutils.dicom import dicom_utils
from epsutils.gcs import gcs_utils
from epsutils.image import image_utils


class GradientFracturesDatasetHelper(BaseDatasetHelper):
    def __init__(self,
                 gcs_chest_images_file,
                 gcs_fractures_file,
                 images_dir=None,
                 dir_prefix_to_remove=None,
                 seed=None):
        super().__init__(gcs_chest_images_file=gcs_chest_images_file,
                         gcs_fractures_file=gcs_fractures_file,
                         images_dir=images_dir,
                         dir_prefix_to_remove=dir_prefix_to_remove,
                         seed=seed)

    def _load_dataset(self, *args, **kwargs):
        # Store params.
        self.__gcs_chest_images_file = kwargs["gcs_chest_images_file"] if "gcs_chest_images_file" in kwargs else next((arg for arg in args if arg == "gcs_chest_images_file"), None)
        self.__gcs_fractures_file = kwargs["gcs_fractures_file"] if "gcs_fractures_file" in kwargs else next((arg for arg in args if arg == "gcs_fractures_file"), None)
        self.__images_dir = kwargs["images_dir"] if "images_dir" in kwargs else next((arg for arg in args if arg == "images_dir"), None)
        self.__dir_prefix_to_remove = kwargs["dir_prefix_to_remove"] if "dir_prefix_to_remove" in kwargs else next((arg for arg in args if arg == "dir_prefix_to_remove"), None)
        self.__seed = kwargs["seed"] if "seed" in kwargs else next((arg for arg in args if arg == "seed"), None)

        # Download chest images file.
        print("Downloading chest images file")
        gcs_data = gcs_utils.split_gcs_uri(self.__gcs_chest_images_file)
        content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

        # Generate a list of chest images.
        print("Generating a list of chest images")
        df = pd.read_csv(StringIO(content), header=None, sep=';')
        chest_images = set(df[0])

        # Download fractures file.
        print("Downloading fractures file")
        gcs_data = gcs_utils.split_gcs_uri(self.__gcs_fractures_file)
        content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])

        # Add frontal images.
        print("Reading fractures file")
        data = []
        lines = content.splitlines()
        for line in lines:
            values = json.loads(line)

            image_path = values["image_path"]
            if image_path not in chest_images:
                continue
            image_path = os.path.relpath(image_path, self.__dir_prefix_to_remove) if self.__dir_prefix_to_remove else image_path
            image_path = os.path.join(self.__images_dir, image_path) if self.__images_dir else image_path

            labels = values["labels"]
            assert labels == ["Fracture (Recent)"] or labels == []
            data.append({"image_path": image_path, "label": 1 if labels == ["Fracture (Recent)"] else 0})

        # Popullate the dataset.
        print("Populating the dataset")
        self.__pandas_full_dataset = pd.DataFrame(data)
        print(f"Full dataset size: {len(self.__pandas_full_dataset)}")
        num_fractures = self.__pandas_full_dataset[self.__pandas_full_dataset['label'] == 1].shape[0]
        num_non_fractures = self.__pandas_full_dataset[self.__pandas_full_dataset['label'] == 0].shape[0]
        print(f"Num fractures: {num_fractures}")
        print(f"Num non-fractures: {num_non_fractures}")

        # Generate splits.
        print("Generating splits")
        self.__pandas_train_dataset, temp = train_test_split(self.__pandas_full_dataset, test_size=0.2, random_state=self.__seed)
        self.__pandas_validation_dataset, self.__pandas_test_dataset = train_test_split(temp, test_size=0.5, random_state=self.__seed)
        print(f"Train dataset size: {len(self.__pandas_train_dataset)}")
        print(f"Validation dataset size: {len(self.__pandas_validation_dataset)}")
        print(f"Test dataset size: {len(self.__pandas_test_dataset)}")

        # Create Torch datasets.
        print("Creating Torch datasets")
        self.__torch_train_dataset = GradientFrontalLateralTorchDataset(pandas_dataframe=self.__pandas_train_dataset)
        self.__torch_validation_dataset = GradientFrontalLateralTorchDataset(pandas_dataframe=self.__pandas_validation_dataset)
        self.__torch_test_dataset = GradientFrontalLateralTorchDataset(pandas_dataframe=self.__pandas_test_dataset)

    def get_pil_image(self, item):
        raise NotImplementedError("Not implemented")

    def get_torch_image(self, item, processor):
        raise NotImplementedError("Not implemented")

    def get_labels(self):
        return ["Non-fracture", "Fracture"]

    def get_ids_to_labels(self):
        return {0: "Non-fracture", 1: "Fracture"}

    def get_labels_to_ids(self):
        return {"Non-fracture": 0, "Fracture": 1}

    def get_torch_label(self, item):
        raise NotImplementedError("Not implemented")

    def get_pandas_full_dataset(self):
        return self.__pandas_full_dataset

    def get_hugging_face_train_dataset(self):
        return datasets.Dataset.from_pandas(self.__pandas_train_dataset)

    def get_hugging_face_validation_dataset(self):
        return datasets.Dataset.from_pandas(self.__pandas_validation_dataset)

    def get_hugging_face_test_dataset(self):
        return datasets.Dataset.from_pandas(self.__pandas_test_dataset)

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
                                 persistent_workers=True)
        return data_loader


class GradientFrontalLateralTorchDataset(Dataset):
    def __init__(self, pandas_dataframe):
        self.__pandas_dataframe = pandas_dataframe

        self.__transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __getitem__(self, idx):
        try:
            item = self.__pandas_dataframe.iloc[idx]
            image_path = item["image_path"]

            # Download image from GCS or load locally.
            if gcs_utils.is_gcs_uri(image_path):
                gcs_data = gcs_utils.split_gcs_uri(image_path)
                content = BytesIO(gcs_utils.download_file_as_bytes(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"]))
            else:
                content = image_path

            # Convert to PIL image.
            image = dicom_utils.get_dicom_image(content, custom_windowing_parameters={"window_center": 0, "window_width": 0})
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
            image = image_utils.numpy_array_to_pil_image(image, convert_to_uint8=False, convert_to_rgb=False)

            # Transform and convert to tensor.
            image = self.__transform(image)

            if torch.any(torch.isnan(image)):
                print(f"ERROR: There are NaN values in the tensor generated from {item['image_path']}")

            # Get labels.
            labels = torch.tensor(item["label"]).float()

            return image, labels

        except Exception as e:
            print(f"Error: {str(e)}   File: {item['image_path']}   Item: {item}")
            raise

    def __len__(self):
        return len(self.__pandas_dataframe)
