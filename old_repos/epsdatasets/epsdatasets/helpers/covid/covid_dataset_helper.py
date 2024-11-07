import os

import datasets
import numpy as np
import pandas as pd
import torch
from natsort import natsorted
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from epsdatasets.helpers.base.base_dataset_helper import BaseDatasetHelper


class CovidDatasetHelper(BaseDatasetHelper):
    def __init__(self, dataset_path, seed=None):
        super().__init__(dataset_path=dataset_path, seed=seed)

    def _load_dataset(self, *args, **kwargs):
        self.__dataset_path = kwargs["dataset_path"] if "dataset_path" in kwargs else next((arg for arg in args if arg == "dataset_path"), None)
        self.__seed = kwargs["seed"] if "seed" in kwargs else next((arg for arg in args if arg == "seed"), None)

        self.__generate_full_dataset()
        self.__generate_splits()
        self.__create_torch_datasets()

    def get_max_depth(self):
        return self.__max_depth

    def get_pil_image(self, item, normalization_depth=None):
        images = [Image.open(os.path.join(item["dir"], image_file)).convert("RGB") for image_file in item["image_files"]]

        if normalization_depth is not None:
            num_images = len(images)
            assert(num_images > 0 and normalization_depth >= num_images)
            num_missing = normalization_depth - num_images
            empty_image = np.array(images[0])
            empty_image.fill(0)
            empty_image = Image.fromarray(empty_image)
            front_pad = (num_missing + 1) // 2
            end_pad = num_missing // 2
            images = [empty_image] * front_pad + images + [empty_image] * end_pad

        return images

    def get_torch_image(self, item, transform, normalization_depth=None):
        images = self.get_pil_image(item, normalization_depth)
        tensors = [transform(image) for image in images]
        stacked_tensor = torch.stack(tensors)
        # Instead of the tensor shape (num_slices, num_channels, image_height, image_width),
        # which is obtained by stacking the tensors, the model requires the following shape:
        # (num_channels, num_slices, image_height, image_width), which is obtained by
        # premuting the dimensions.
        stacked_tensor = stacked_tensor.permute(1, 0, 2, 3)
        return stacked_tensor

    def get_labels(self):
        return self.__labels

    def get_ids_to_labels(self):
        return self.__ids_to_labels

    def get_labels_to_ids(self):
        return self.__labels_to_ids

    def get_torch_label(self, item):
        id = self.__labels_to_ids[item["label"]]
        return torch.tensor(id, dtype=torch.int64)

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
        data_loader = DataLoader(self.__torch_train_dataset, collate_fn=collate_function,
                                 batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return data_loader

    def get_torch_validation_data_loader(self, collate_function, batch_size, num_workers):
        data_loader = DataLoader(self.__torch_validation_dataset, collate_fn=collate_function,
                                 batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return data_loader

    def get_torch_test_data_loader(self, collate_function, batch_size, num_workers):
        data_loader = DataLoader(self.__torch_test_dataset, collate_fn=collate_function,
                                 batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return data_loader

    def __generate_full_dataset(self):
        self.__labels = []
        for item in os.listdir(self.__dataset_path):
            if os.path.isdir(os.path.join(self.__dataset_path, item)):
                self.__labels.append(item)

        self.__labels_to_ids = {label: i for i, label in enumerate(self.__labels)}
        self.__ids_to_labels = {i: label for i, label in enumerate(self.__labels)}

        self.__pandas_full_dataset = pd.DataFrame(columns=["dir", "image_files", "label"])

        for label in self.__labels:
            label_dir = os.path.join(self.__dataset_path, label)

            for item in os.listdir(label_dir):
                patient_dir = os.path.join(label_dir, item)

                if not os.path.isdir(patient_dir):
                    continue

                image_files = self.__get_all_image_files(patient_dir)
                self.__pandas_full_dataset = self.__pandas_full_dataset._append({"dir": patient_dir, "image_files": image_files, "label": label}, ignore_index=True)

        self.__max_depth = self.__pandas_full_dataset["image_files"].apply(len).max()

    def __get_all_image_files(self, dir, sorted=True):
        image_files = [file for file in os.listdir(dir) if file.endswith(".png")]
        if sorted:
            image_files = natsorted(image_files)
        return image_files

    def __generate_splits(self):
        self.__pandas_train_dataset, temp = train_test_split(self.__pandas_full_dataset, test_size=0.2, random_state=self.__seed)
        self.__pandas_validation_dataset, self.__pandas_test_dataset = train_test_split(temp, test_size=0.5, random_state=self.__seed)

    def __create_torch_datasets(self):
        self.__torch_train_dataset = CovidTorchDataset(pandas_dataframe=self.__pandas_train_dataset)
        self.__torch_validation_dataset = CovidTorchDataset(pandas_dataframe=self.__pandas_validation_dataset)
        self.__torch_test_dataset = CovidTorchDataset(pandas_dataframe=self.__pandas_test_dataset)


class CovidTorchDataset(Dataset):
    def __init__(self, pandas_dataframe):
        self.__pandas_dataframe = pandas_dataframe

    def __getitem__(self, idx):
        item = self.__pandas_dataframe.iloc[idx]
        return item

    def __len__(self):
        return len(self.__pandas_dataframe)
