import json
import os
from multiprocessing import Pool

import datasets
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from epsdatasets.helpers.base.base_dataset_helper import BaseDatasetHelper
from epsutils.labels.grouped_labels_manager import GroupedLabelsManager


class FawkesDatasetHelper(BaseDatasetHelper):
    def __init__(self, labeled_data_file, grouped_labels_file, volume_depth_threshold=None, use_half_precision=False, seed=None):
        super().__init__(
            labeled_data_file=labeled_data_file, grouped_labels_file=grouped_labels_file,
            volume_depth_threshold=volume_depth_threshold, use_half_precision=use_half_precision,
            seed=seed)

    def _load_dataset(self, *args, **kwargs):
        self.__labeled_data_file = kwargs["labeled_data_file"] if "labeled_data_file" in kwargs else next((arg for arg in args if arg == "labeled_data_file"), None)
        self.__grouped_labels_file = kwargs["grouped_labels_file"] if "grouped_labels_file" in kwargs else next((arg for arg in args if arg == "grouped_labels_file"), None)
        self.__volume_depth_threshold = kwargs["volume_depth_threshold"] if "volume_depth_threshold" in kwargs else next((arg for arg in args if arg == "volume_depth_threshold"), None)
        self.__use_half_precision = kwargs["use_half_precision"] if "use_half_precision" in kwargs else next((arg for arg in args if arg == "use_half_precision"), None)
        self.__seed = kwargs["seed"] if "seed" in kwargs else next((arg for arg in args if arg == "seed"), None)

        self.__root_dir = os.path.dirname(self.__labeled_data_file)

        self.__dtype = torch.float16 if self.__use_half_precision else torch.float32

        print("Creating grouped labels manager")
        self.__create_grouped_labels_manager()

        print("Generating full dataset")
        self.__generate_full_dataset()

        print("Generating splits")
        self.__generate_splits()

        print("Creating torch datasets")
        self.__create_torch_datasets()

    def get_max_depth(self):
        return self.__max_depth

    def get_pil_image(self, item, normalization_depth=None):
        image_array = sitk.GetArrayFromImage(sitk.ReadImage(item["volume_file"]))

        images = []
        for i in range(image_array.shape[0]):
            nifti_image = image_array[i, :, :]
            max = np.max(nifti_image)
            min = np.min(nifti_image)
            if (max > min):
                nifti_image = ((nifti_image - min) / (max - min) * 255).astype(np.uint8)
            else:
                nifti_image.fill(0)
                nifti_image = nifti_image.astype(np.uint8)
            image = Image.fromarray(nifti_image).convert("RGB")
            images.append(image)

        if normalization_depth is not None:
            num_images = len(images)

            if num_images == 0:
                raise ValueError("No images in the volume")

            if normalization_depth < num_images:
                raise ValueError(f"Normalization depth ({normalization_depth}) < number of images in the volume ({num_images}), "
                                 f"it should be greater or equal to number of images")

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
        tensors = [transform(image).to(self.__dtype) for image in images]
        stacked_tensor = torch.stack(tensors)
        # Instead of the tensor shape (num_slices, num_channels, image_height, image_width),
        # which is obtained by stacking the tensors, the model requires the following shape:
        # (num_channels, num_slices, image_height, image_width), which is obtained by
        # premuting the dimensions.
        stacked_tensor = stacked_tensor.permute(1, 0, 2, 3)
        return stacked_tensor

    def get_labels(self):
        return self.__grouped_labels_manager.get_groups()

    def get_ids_to_labels(self):
        return self.__grouped_labels_manager.get_ids_to_groups()

    def get_labels_to_ids(self):
        return self.__grouped_labels_manager.get_groups_to_ids()

    def get_torch_label(self, item):
        return torch.tensor(
            self.__grouped_labels_manager.encoded_string_to_encoded_list(item["label"]), dtype=self.__dtype)

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

    def __create_grouped_labels_manager(self):
        with open(self.__grouped_labels_file, "r") as json_file:
            grouped_labels = json.load(json_file)

        self.__grouped_labels_manager = GroupedLabelsManager(grouped_labels)  # TODO: Set allow_duplicate_labels to False.

    def _append_num_slices_task(self, item):
        volume_file = os.path.join(self.__root_dir, item["volume_file"])
        label = item["label"]
        img = nib.load(volume_file, mmap=True)
        num_slices = img.shape[2]
        return {"volume_file": volume_file, "label": label, "num_slices": num_slices}

    def __generate_full_dataset(self):
        with open(self.__labeled_data_file, "r") as json_file:
            labeled_data = json.load(json_file)

        print("Scanning volumes for number of slices (this may take a while)")
        with Pool() as pool:
            labeled_data = pool.map(self._append_num_slices_task, labeled_data)

        if self.__volume_depth_threshold is not None:
            ignored = [item for item in labeled_data if item["num_slices"] >= self.__volume_depth_threshold]
            labeled_data = [item for item in labeled_data if item["num_slices"] < self.__volume_depth_threshold]
            if ignored:
                print(f"The following {len(ignored)} volumes with number of slices >= {self.__volume_depth_threshold} will be ignored:")
                for item in ignored:
                    print(f"{item['volume_file']} ({item['num_slices']} slices)")

        self.__max_depth = 0
        for item in labeled_data:
            if item["num_slices"] > self.__max_depth:
                self.__max_depth = item["num_slices"]

        self.__pandas_full_dataset = pd.DataFrame(labeled_data)
        self.__pandas_full_dataset = self.__pandas_full_dataset.sort_values(by="volume_file")

        print(f"Full dataset size: {len(self.__pandas_full_dataset)}")

    def __generate_splits(self):
        self.__pandas_train_dataset, temp = train_test_split(self.__pandas_full_dataset, test_size=0.2, random_state=self.__seed)
        self.__pandas_validation_dataset, self.__pandas_test_dataset = train_test_split(temp, test_size=0.5, random_state=self.__seed)

        print(f"Train dataset size: {len(self.__pandas_train_dataset)}")
        print(f"Validation dataset size: {len(self.__pandas_validation_dataset)}")
        print(f"Test dataset size: {len(self.__pandas_test_dataset)}")

    def __create_torch_datasets(self):
        self.__torch_train_dataset = FawkesTorchDataset(pandas_dataframe=self.__pandas_train_dataset)
        self.__torch_validation_dataset = FawkesTorchDataset(pandas_dataframe=self.__pandas_validation_dataset)
        self.__torch_test_dataset = FawkesTorchDataset(pandas_dataframe=self.__pandas_test_dataset)


class FawkesTorchDataset(Dataset):
    def __init__(self, pandas_dataframe):
        self.__pandas_dataframe = pandas_dataframe

    def __getitem__(self, idx):
        item = self.__pandas_dataframe.iloc[idx]
        return item

    def __len__(self):
        return len(self.__pandas_dataframe)
