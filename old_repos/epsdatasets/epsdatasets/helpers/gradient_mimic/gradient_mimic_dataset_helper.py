import os
from io import BytesIO

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


class GradientMimicDatasetHelper(BaseDatasetHelper):
    def __init__(self,
                 gradient_data_gcs_bucket_name,
                 gradient_data_gcs_dir,
                 gradient_images_gcs_bucket_name,
                 gradient_images_gcs_dir,
                 mimic_gcs_bucket_name,
                 mimic_gcs_dir,
                 exclude_file_name=None,
                 seed=None):
        super().__init__(gradient_data_gcs_bucket_name=gradient_data_gcs_bucket_name,
                         gradient_data_gcs_dir=gradient_data_gcs_dir,
                         gradient_images_gcs_bucket_name=gradient_images_gcs_bucket_name,
                         gradient_images_gcs_dir=gradient_images_gcs_dir,
                         mimic_gcs_bucket_name=mimic_gcs_bucket_name,
                         mimic_gcs_dir=mimic_gcs_dir,
                         exclude_file_name=exclude_file_name,
                         seed=seed)

    def _load_dataset(self, *args, **kwargs):
        # Store params.
        self.__gradient_data_gcs_bucket_name = kwargs["gradient_data_gcs_bucket_name"] if "gradient_data_gcs_bucket_name" in kwargs else next((arg for arg in args if arg == "gradient_data_gcs_bucket_name"), None)
        self.__gradient_data_gcs_dir = kwargs["gradient_data_gcs_dir"] if "gradient_data_gcs_dir" in kwargs else next((arg for arg in args if arg == "gradient_data_gcs_dir"), None)
        self.__gradient_images_gcs_bucket_name = kwargs["gradient_images_gcs_bucket_name"] if "gradient_images_gcs_bucket_name" in kwargs else next((arg for arg in args if arg == "gradient_images_gcs_bucket_name"), None)
        self.__gradient_images_gcs_dir = kwargs["gradient_images_gcs_dir"] if "gradient_images_gcs_dir" in kwargs else next((arg for arg in args if arg == "gradient_images_gcs_dir"), None)
        self.__mimic_gcs_bucket_name = kwargs["mimic_gcs_bucket_name"] if "mimic_gcs_bucket_name" in kwargs else next((arg for arg in args if arg == "mimic_gcs_bucket_name"), None)
        self.__mimic_gcs_dir = kwargs["mimic_gcs_dir"] if "mimic_gcs_dir" in kwargs else next((arg for arg in args if arg == "mimic_gcs_dir"), None)
        self.__exclude_file_name = kwargs["exclude_file_name"] if "exclude_file_name" in kwargs else next((arg for arg in args if arg == "exclude_file_name"), None)
        self.__seed = kwargs["seed"] if "seed" in kwargs else next((arg for arg in args if arg == "seed"), None)

        # Download a list of Mimic files.
        print("Downloading a list of Mimic files")
        image_file_names_file = os.path.join(self.__mimic_gcs_dir, "IMAGE_FILENAMES")
        content = gcs_utils.download_file_as_string(gcs_bucket_name=self.__mimic_gcs_bucket_name, gcs_file_name=image_file_names_file)
        mimic_file_names = content.split("\n")
        mimic_file_names = [os.path.join(self.__mimic_gcs_dir, file_name) for file_name in mimic_file_names]

        # Read a list of Mimic files.
        print("Reading a list of Mimic files")
        rows = []
        for file_name in mimic_file_names:
            rows.append({"file_path": file_name, "gcs_bucket_name": self.__mimic_gcs_bucket_name, "dataset": "mimic", "label": "chest"})

        # Get a list of Gradient files.
        print("Getting a list of Gradient files in the bucket")
        gradient_file_names = gcs_utils.list_files(gcs_bucket_name=self.__gradient_data_gcs_bucket_name, gcs_dir=self.__gradient_data_gcs_dir)
        gradient_file_names = [file_name for file_name in gradient_file_names if file_name.endswith(".txt")]

        if self.__exclude_file_name is not None:
            df = pd.read_csv(self.__exclude_file_name, delimiter=";", header=None)
            files_to_exclude = df.iloc[:, 1].tolist()
            files_to_exclude = set(files_to_exclude)
            print(f"Num Gradient files that will be excluded: {len(files_to_exclude)}")
            print(f"All Gradient files before removal: {len(gradient_file_names)}")
            gradient_file_names = [file_name for file_name in gradient_file_names if file_name not in files_to_exclude]
            print(f"All Gradient files after removal: {len(gradient_file_names)}")

        # Download the same number of Gradient files as in the Mimic dataset so that we have a balanced dataset.
        num_files_to_get = len(rows)
        gradient_file_names = [os.path.basename(file_name) for file_name in gradient_file_names[:num_files_to_get]]

        # Read a list of Gradient files.
        print("Reading a list of Gradient files")
        for file_name in gradient_file_names:
            dicom_file_name = os.path.join(self.__gradient_images_gcs_dir, file_name.replace("_", "/").replace(".txt", ".dcm"))
            rows.append({"file_path": dicom_file_name, "gcs_bucket_name": self.__gradient_images_gcs_bucket_name, "dataset": "gradient", "label": "other"})

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
        self.__torch_train_dataset = GradientMimicTorchDataset(pandas_dataframe=self.__pandas_train_dataset)
        self.__torch_validation_dataset = GradientMimicTorchDataset(pandas_dataframe=self.__pandas_validation_dataset)
        self.__torch_test_dataset = GradientMimicTorchDataset(pandas_dataframe=self.__pandas_test_dataset)

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


class GradientMimicTorchDataset(Dataset):
    def __init__(self, pandas_dataframe):
        self.__pandas_dataframe = pandas_dataframe

        self.__transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __getitem__(self, idx):
        try:
            item = self.__pandas_dataframe.iloc[idx]

            # Download file.
            content = gcs_utils.download_file_as_bytes(gcs_bucket_name=item["gcs_bucket_name"], gcs_file_name=item["file_path"])

            # Convert to PIL image.
            if item["dataset"] == "gradient":
                image = dicom_utils.get_dicom_image(BytesIO(content), custom_windowing_parameters={"window_center": 0, "window_width": 0})
                image = image.astype(np.float32)
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                image = Image.fromarray(image)
            else:
                image = Image.open(BytesIO(content))

            # Transform image and conver to tensor.
            image = self.__transform(image)

            # Get label.
            label = 1 if item["label"] == "chest" else 0
            label = torch.tensor([label]).float()

            return image, label

        except Exception as e:
            print(f"Error: {str(e)}   File: {item['file_path']}   Item: {item}")
            raise

    def __len__(self):
        return len(self.__pandas_dataframe)


if __name__ == "__main__":
    helper = GradientMimicDatasetHelper(gradient_data_gcs_bucket_name="gradient-crs",
                                        gradient_data_gcs_dir="16AG02924",
                                        gradient_images_gcs_bucket_name="epsilon-data-us-central1",
                                        gradient_images_gcs_dir="GRADIENT-DATABASE/CR/16AG02924",
                                        mimic_gcs_bucket_name="epsilonlabs-filestore",
                                        mimic_gcs_dir="mimic2-dicom/mimic-cxr-jpg-2.1.0.physionet.org",
                                        seed=42)

    train_dataset = helper.get_torch_train_dataset()

    for i in range(10):
        item, label = train_dataset.__getitem__(i)
        print(label)
