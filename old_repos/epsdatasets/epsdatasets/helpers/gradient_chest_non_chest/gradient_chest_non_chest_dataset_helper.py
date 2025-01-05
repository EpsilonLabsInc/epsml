import os
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from epsdatasets.helpers.base.base_dataset_helper import BaseDatasetHelper
from epsutils.dicom import dicom_utils
from epsutils.gcs import gcs_utils


class GradientChestNonChestDatasetHelper(BaseDatasetHelper):
    def __init__(self,
                 chest_data_gcs_bucket_name,
                 chest_data_gcs_dir,
                 chest_images_gcs_bucket_name,
                 chest_images_gcs_dir,
                 non_chest_data_gcs_bucket_name,
                 non_chest_data_gcs_dir,
                 non_chest_images_gcs_bucket_name,
                 non_chest_images_gcs_dir,
                 chest_exclude_file_name=None,
                 non_chest_exclude_file_name=None,
                 seed=None):
        super().__init__(chest_data_gcs_bucket_name=chest_data_gcs_bucket_name,
                         chest_data_gcs_dir=chest_data_gcs_dir,
                         chest_images_gcs_bucket_name=chest_images_gcs_bucket_name,
                         chest_images_gcs_dir=chest_images_gcs_dir,
                         non_chest_data_gcs_bucket_name=non_chest_data_gcs_bucket_name,
                         non_chest_data_gcs_dir=non_chest_data_gcs_dir,
                         non_chest_images_gcs_bucket_name=non_chest_images_gcs_bucket_name,
                         non_chest_images_gcs_dir=non_chest_images_gcs_dir,
                         chest_exclude_file_name=chest_exclude_file_name,
                         non_chest_exclude_file_name=non_chest_exclude_file_name,
                         seed=seed)

    def _load_dataset(self, *args, **kwargs):
        # Store params.
        self.__chest_data_gcs_bucket_name = kwargs["chest_data_gcs_bucket_name"] if "chest_data_gcs_bucket_name" in kwargs else next((arg for arg in args if arg == "chest_data_gcs_bucket_name"), None)
        self.__chest_data_gcs_dir = kwargs["chest_data_gcs_dir"] if "chest_data_gcs_dir" in kwargs else next((arg for arg in args if arg == "chest_data_gcs_dir"), None)
        self.__chest_images_gcs_bucket_name = kwargs["chest_images_gcs_bucket_name"] if "chest_images_gcs_bucket_name" in kwargs else next((arg for arg in args if arg == "chest_images_gcs_bucket_name"), None)
        self.__chest_images_gcs_dir = kwargs["chest_images_gcs_dir"] if "chest_images_gcs_dir" in kwargs else next((arg for arg in args if arg == "chest_images_gcs_dir"), None)
        self.__non_chest_data_gcs_bucket_name = kwargs["non_chest_data_gcs_bucket_name"] if "non_chest_data_gcs_bucket_name" in kwargs else next((arg for arg in args if arg == "non_chest_data_gcs_bucket_name"), None)
        self.__non_chest_data_gcs_dir = kwargs["non_chest_data_gcs_dir"] if "non_chest_data_gcs_dir" in kwargs else next((arg for arg in args if arg == "non_chest_data_gcs_dir"), None)
        self.__non_chest_images_gcs_bucket_name = kwargs["non_chest_images_gcs_bucket_name"] if "non_chest_images_gcs_bucket_name" in kwargs else next((arg for arg in args if arg == "non_chest_images_gcs_bucket_name"), None)
        self.__non_chest_images_gcs_dir = kwargs["non_chest_images_gcs_dir"] if "non_chest_images_gcs_dir" in kwargs else next((arg for arg in args if arg == "non_chest_images_gcs_dir"), None)
        self.__chest_exclude_file_name = kwargs["chest_exclude_file_name"] if "chest_exclude_file_name" in kwargs else next((arg for arg in args if arg == "chest_exclude_file_name"), None)
        self.__non_chest_exclude_file_name = kwargs["non_chest_exclude_file_name"] if "non_chest_exclude_file_name" in kwargs else next((arg for arg in args if arg == "non_chest_exclude_file_name"), None)
        self.__seed = kwargs["seed"] if "seed" in kwargs else next((arg for arg in args if arg == "seed"), None)

        # Get a list of non-chest files.
        print("Getting a list of non-chest files in the bucket")
        non_chest_file_names = gcs_utils.list_files(gcs_bucket_name=self.__non_chest_data_gcs_bucket_name, gcs_dir=self.__non_chest_data_gcs_dir)
        non_chest_file_names = [file_name for file_name in non_chest_file_names if file_name.endswith(".txt")]

        if self.__non_chest_exclude_file_name is not None:
            df = pd.read_csv(self.__non_chest_exclude_file_name, delimiter=";", header=None)
            files_to_exclude = df.iloc[:, 1].tolist()
            files_to_exclude = set(files_to_exclude)
            print(f"Num non-chest files that will be excluded: {len(files_to_exclude)}")
            print(f"All non-chest files before removal: {len(non_chest_file_names)}")
            non_chest_file_names = [file_name for file_name in non_chest_file_names if file_name not in files_to_exclude]
            print(f"All non-chest files after removal: {len(non_chest_file_names)}")

        # Get a list of chest files.
        print("Getting a list of chest files in the bucket")
        chest_file_names = gcs_utils.list_files(gcs_bucket_name=self.__chest_data_gcs_bucket_name, gcs_dir=self.__chest_data_gcs_dir)
        chest_file_names = [file_name for file_name in chest_file_names if file_name.endswith(".txt")]

        if self.__chest_exclude_file_name is not None:
            df = pd.read_csv(self.__chest_exclude_file_name, delimiter=";", header=None)
            files_to_exclude = df.iloc[:, 1].tolist()
            files_to_exclude = set(files_to_exclude)
            print(f"Num chest files that will be excluded: {len(files_to_exclude)}")
            print(f"All chest files before removal: {len(chest_file_names)}")
            chest_file_names = [file_name for file_name in chest_file_names if file_name not in files_to_exclude]
            print(f"All chest files after removal: {len(chest_file_names)}")

        # Download the same number of chest and non-chest files so that we have a balanced dataset.
        num_files_to_get = 300000
        non_chest_file_names = [os.path.basename(file_name) for file_name in non_chest_file_names[:num_files_to_get]]
        chest_file_names = [os.path.basename(file_name) for file_name in chest_file_names[:num_files_to_get]]
        rows = []

        # Read a list of non-chest files.
        print("Reading a list of non-chest files")
        for file_name in non_chest_file_names:
            dicom_file_name = os.path.join(self.__non_chest_images_gcs_dir, file_name.replace("_", "/").replace(".txt", ".dcm"))
            rows.append({"file_path": dicom_file_name, "gcs_bucket_name": self.__non_chest_images_gcs_bucket_name, "label": "other"})

        # Read a list of chest files.
        print("Reading a list of chest files")
        for file_name in chest_file_names:
            dicom_file_name = os.path.join(self.__chest_images_gcs_dir, file_name.replace("_", "/").replace(".txt", ".dcm"))
            rows.append({"file_path": dicom_file_name, "gcs_bucket_name": self.__chest_images_gcs_bucket_name, "label": "chest"})

        # Popullate the dataset.
        print("Populating the dataset")
        self.__pandas_full_dataset = pd.DataFrame(rows)
        print(f"Full dataset size: {len(self.__pandas_full_dataset)}")

        # Generate splits.
        print("Generating splits")
        self.__pandas_train_dataset, temp = train_test_split(self.__pandas_full_dataset, test_size=0.2, random_state=self.__seed)
        self.__pandas_validation_dataset, self.__pandas_test_dataset = train_test_split(temp, test_size=0.5, random_state=self.__seed)
        print(f"Train dataset size: {len(self.__pandas_train_dataset)}")
        print(f"Validation dataset size: {len(self.__pandas_validation_dataset)}")
        print(f"Test dataset size: {len(self.__pandas_test_dataset)}")

        # Create Torch datasets.
        print("Creating Torch datasets")
        self.__torch_train_dataset = GradientChestNonChestTorchDataset(pandas_dataframe=self.__pandas_train_dataset)
        self.__torch_validation_dataset = GradientChestNonChestTorchDataset(pandas_dataframe=self.__pandas_validation_dataset)
        self.__torch_test_dataset = GradientChestNonChestTorchDataset(pandas_dataframe=self.__pandas_test_dataset)

    def get_pil_image(self, item):
        raise NotImplementedError("Not implemented")

    def get_torch_image(self, item, processor):
        raise NotImplementedError("Not implemented")

    def get_labels(self):
        raise NotImplementedError("Not implemented")

    def get_ids_to_labels(self):
        raise NotImplementedError("Not implemented")

    def get_labels_to_ids(self):
        raise NotImplementedError("Not implemented")

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


class GradientChestNonChestTorchDataset(Dataset):
    def __init__(self, pandas_dataframe):
        self.__pandas_dataframe = pandas_dataframe

        self.__transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __getitem__(self, idx):
        try:
            item = self.__pandas_dataframe.iloc[idx]

            # Download file.
            content = gcs_utils.download_file_as_bytes(gcs_bucket_name=item["gcs_bucket_name"], gcs_file_name=item["file_path"])

            # Convert to PIL image.
            image = dicom_utils.get_dicom_image(BytesIO(content), custom_windowing_parameters={"window_center": 0, "window_width": 0})
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = (image - image.min()) / (image.max() - image.min())
            image = Image.fromarray(image)

            # Transform image and convert to tensor.
            image = self.__transform(image)

            if torch.any(torch.isnan(image)):
                print(f"ERROR: There are NaN values in the tensor generated from {item['file_path'].replace('/', '_').replace('.dcm', '.txt')}")

            # Get label.
            label = 1 if item["label"] == "chest" else 0
            label = torch.tensor([label]).float()

            return image, label

        except Exception as e:
            print(f"Error: {str(e)}   File: {item['file_path']}   Item: {item}")
            raise

    def __len__(self):
        return len(self.__pandas_dataframe)
