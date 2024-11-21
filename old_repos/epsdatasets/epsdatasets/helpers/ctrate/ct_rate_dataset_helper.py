import os

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars
from torch.utils.data import Dataset, DataLoader

from epsdatasets.helpers.base.base_dataset_helper import BaseDatasetHelper
from epsutils.nifti import nifti_utils


class CtRateDatasetHelper(BaseDatasetHelper):
    def __init__(self, training_file, validation_file):
        super().__init__(training_file=training_file, validation_file=validation_file)

        # HuggingFace settings.
        self.__repo_id = "ibrahimhamamci/CT-RATE"
        self.__hf_token = os.getenv('HF_TOKEN')

        # Disable HuggingFace progress bars.
        disable_progress_bars()

    def _load_dataset(self, *args, **kwargs):
        self.__training_file = kwargs["training_file"] if "training_file" in kwargs else next((arg for arg in args if arg == "training_file"), None)
        self.__validation_file = kwargs["validation_file"] if "validation_file" in kwargs else next((arg for arg in args if arg == "validation_file"), None)

        self.__pandas_train_dataset = pd.read_csv(self.__training_file)
        self.__pandas_validation_dataset = pd.read_csv(self.__validation_file)

        self.__torch_train_dataset = CtRateTorchDataset(pandas_dataframe=self.__pandas_train_dataset)
        self.__torch_validation_dataset = CtRateTorchDataset(pandas_dataframe=self.__pandas_validation_dataset)

        self.__labels = list(self.__pandas_train_dataset.columns)
        self.__labels.remove("VolumeName")

    def get_max_depth(self):
        raise NotImplementedError("Method not implemented")

    def get_pil_image(self, item, split, target_image_size=None, normalization_depth=None, sample_slices=False):
        if split not in ["train", "valid"]:
            raise ValueError("Argument split must be either 'train' or 'valid")

        # Obtain path from filename.
        file_name = item["VolumeName"]
        folder1 = file_name.split('_')[0]
        folder2 = file_name.split('_')[1]
        folder = folder1 + '_' + folder2
        folder3 = file_name.split('_')[2]
        subfolder = folder + '_' + folder3
        subfolder = f"dataset/{split}/" + folder + '/' + subfolder

        # Download image.
        hf_hub_download(repo_id=self.__repo_id, repo_type="dataset", token=self.__hf_token, subfolder=subfolder, filename=file_name, cache_dir=".", local_dir=".")

        # Read image.
        nifti_file = os.path.join(subfolder, file_name)
        slices = nifti_utils.nifti_file_to_pil_images(nifti_file=nifti_file,
                                                      source_data_type=np.float64,
                                                      target_data_type=np.float32,
                                                      target_image_size=target_image_size,
                                                      normalization_depth=normalization_depth,
                                                      sample_slices=sample_slices)

        # Delete image from the disk.
        if os.path.exists(nifti_file):
            os.remove(nifti_file)

        return slices

    def get_torch_image(self, item):
        raise NotImplementedError("Method not implemented")

    def get_labels(self):
        return self.__labels

    def get_ids_to_labels(self):
        raise NotImplementedError("Method not implemented")

    def get_labels_to_ids(self):
        raise NotImplementedError("Method not implemented")

    def get_torch_label(self, item):
        vec = [
            item["Medical material"],
            item["Arterial wall calcification"],
            item["Cardiomegaly"],
            item["Pericardial effusion"],
            item["Coronary artery wall calcification"],
            item["Hiatal hernia"],
            item["Lymphadenopathy"],
            item["Emphysema"],
            item["Atelectasis"],
            item["Lung nodule"],
            item["Lung opacity"],
            item["Pulmonary fibrotic sequela"],
            item["Pleural effusion"],
            item["Mosaic attenuation pattern"],
            item["Peribronchial thickening"],
            item["Consolidation"],
            item["Bronchiectasis"],
            item["Interlobular septal thickening"]
        ]

        return torch.tensor(vec)

    def get_pandas_full_dataset(self):
        raise NotImplementedError("Method not implemented")

    def get_hugging_face_train_dataset(self):
        raise NotImplementedError("Method not implemented")

    def get_hugging_face_validation_dataset(self):
        raise NotImplementedError("Method not implemented")

    def get_hugging_face_test_dataset(self):
        raise NotImplementedError("Method not implemented")

    def get_torch_train_dataset(self):
        return self.__torch_train_dataset

    def get_torch_validation_dataset(self):
        return self.__torch_validation_dataset

    def get_torch_test_dataset(self):
        raise NotImplementedError("Method not implemented")

    def get_torch_train_data_loader(self, collate_function, batch_size, num_workers):
        data_loader = DataLoader(self.__torch_train_dataset, collate_fn=collate_function,
                                 batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return data_loader

    def get_torch_validation_data_loader(self, collate_function, batch_size, num_workers):
        data_loader = DataLoader(self.__torch_validation_dataset, collate_fn=collate_function,
                                 batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return data_loader

    def get_torch_test_data_loader(self, collate_function, batch_size, num_workers):
        raise NotImplementedError("Method not implemented")


class CtRateTorchDataset(Dataset):
    def __init__(self, pandas_dataframe):
        self.__pandas_dataframe = pandas_dataframe

    def __getitem__(self, idx):
        item = self.__pandas_dataframe.iloc[idx]
        return item

    def __len__(self):
        return len(self.__pandas_dataframe)
