import os
from io import BytesIO, StringIO

import datasets
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from epsdatasets.helpers.base.base_dataset_helper import BaseDatasetHelper
from epsutils.gcs import gcs_utils
from epsutils.labels import labels_utils

IMAGE_FILE_NAMES_FILE = "IMAGE_FILENAMES"
META_DATA_FILE = "mimic-cxr-2.0.0-metadata.csv.gz"
SPLIT_FILE = "mimic-cxr-2.0.0-split.csv.gz"
CHEXPERT_FILE = "mimic-cxr-2.0.0-chexpert.csv.gz"
NEGBIO_FILE = "mimic-cxr-2.0.0-negbio.csv.gz"


class MimicTwoDatasetHelper(BaseDatasetHelper):
    def __init__(self, gcs_uri, labels_generator="chexpert", binary_label=None):
        super().__init__(gcs_uri=gcs_uri, labels_generator=labels_generator, binary_label=binary_label)

    def _load_dataset(self, *args, **kwargs):
        self.__gcs_uri = kwargs["gcs_uri"] if "gcs_uri" in kwargs else next((arg for arg in args if arg == "gcs_uri"), None)
        self.__labels_generator = kwargs["labels_generator"] if "labels_generator" in kwargs else next((arg for arg in args if arg == "labels_generator"), None)
        self.__binary_label = kwargs["binary_label"] if "binary_label" in kwargs else next((arg for arg in args if arg == "binary_label"), None)

        if self.__labels_generator == "chexpert":
            self.__labels_file = CHEXPERT_FILE
        elif self.__labels_generator == "negbio":
            self.__labels_file = NEGBIO_FILE
        else:
            raise ValueError(f"Unsupported labels generator '{self.__labels_generator}', choose either 'chexpert' or 'negbio'")

        self.__pandas_full_dataset = None
        self.__pandas_train_dataset = None
        self.__pandas_validation_dataset = None
        self.__pandas_test_dataset = None
        self.__torch_train_dataset = None
        self.__torch_validation_dataset = None
        self.__torch_test_dataset = None

        self.__load_data()
        self.__get_label_names()
        self.__generate_dataset()
        self.__create_splits()
        self.__create_torch_datasets()

    def get_pil_image(self, item):
        try:
            file_name = item["file_name"]
            gcs_data = gcs_utils.split_gcs_uri(file_name)
            image = BytesIO(gcs_utils.download_file_as_bytes(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"]))
            image = Image.open(image)
            return image

        except Exception as e:
            print(f"Error downloading {item['file_name']}: {str(e)}")
            raise

    def get_torch_image(self, item, processor):
        raise NotImplementedError("Not implemented")

    def get_labels(self):
        return self.__all_labels

    def get_ids_to_labels(self):
        return self.__ids_to_labels

    def get_labels_to_ids(self):
        return self.__labels_to_ids

    def get_torch_label(self, item):
        return torch.tensor(labels_utils.to_multi_hot_encoding(item["labels"], self.__all_labels))

    def get_pandas_full_dataset(self):
        return self.__pandas_full_dataset

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
                                 persistent_workers=True)
        return data_loader

    def __load_data(self):
        print("Downloading image file names file")
        image_file_names_file = os.path.join(self.__gcs_uri, IMAGE_FILE_NAMES_FILE)
        gcs_data = gcs_utils.split_gcs_uri(image_file_names_file)
        content = gcs_utils.download_file_as_string(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])
        self.__image_file_names_dataset = pd.read_csv(StringIO(content), compression="gzip", header=None)
        self.__image_file_names_dataset.columns = ["file_name"]

        print("Downloading labels file")
        labels_file = os.path.join(self.__gcs_uri, self.__labels_file)
        gcs_data = gcs_utils.split_gcs_uri(labels_file)
        content = gcs_utils.download_file_as_bytes(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])
        self.__labels_dataset = pd.read_csv(BytesIO(content), compression="gzip")

        print("Downloading meta data file")
        meta_data_file = os.path.join(self.__gcs_uri, META_DATA_FILE)
        gcs_data = gcs_utils.split_gcs_uri(meta_data_file)
        content = gcs_utils.download_file_as_bytes(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])
        self.__meta_data_dataset = pd.read_csv(BytesIO(content), compression="gzip")

        print("Downloading split file")
        split_file = os.path.join(self.__gcs_uri, SPLIT_FILE)
        gcs_data = gcs_utils.split_gcs_uri(split_file)
        content = gcs_utils.download_file_as_bytes(gcs_bucket_name=gcs_data["gcs_bucket_name"], gcs_file_name=gcs_data["gcs_path"])
        self.__splits_dataset = pd.read_csv(BytesIO(content), compression="gzip")

    def __get_label_names(self):
        if self.__binary_label:
            label_names = [self.__binary_label]
            print(f"Using the following label names: {label_names}")
        else:
            columns_list = self.__labels_dataset.columns.values.tolist()
            label_names = [column for column in columns_list if column not in ["subject_id", "study_id"]]
            print(f"Found the following label names: {label_names}")

        self.__all_labels = label_names
        self.__labels_to_ids = {label: i for i, label in enumerate(self.__all_labels)}
        self.__ids_to_labels = {i: label for i, label in enumerate(self.__all_labels)}

    def __generate_dataset(self):
        print("Generating dataset")

        # Create dataset.
        self.__pandas_full_dataset = self.__image_file_names_dataset.copy(deep=True)
        self.__pandas_full_dataset["subject_id"] = self.__pandas_full_dataset["file_name"].apply(lambda x: x.split('/')[2][1:]).astype(int)  # Remove 'p' prefix from subject ID.
        self.__pandas_full_dataset["study_id"] = self.__pandas_full_dataset["file_name"].apply(lambda x: x.split('/')[3][1:]).astype(int)  # Remove 's' prefix from study ID.
        self.__pandas_full_dataset["dicom_id"] = self.__pandas_full_dataset["file_name"].apply(lambda x: x.split('/')[4].replace(".jpg", ""))  # Remove .jpg extension.

        # Convert all subject and study IDs from string to int for correct comparison.
        self.__labels_dataset["subject_id"] = self.__labels_dataset["subject_id"].astype(int)
        self.__labels_dataset["study_id"] = self.__labels_dataset["study_id"].astype(int)
        self.__splits_dataset["subject_id"] = self.__splits_dataset["subject_id"].astype(int)
        self.__splits_dataset["study_id"] = self.__splits_dataset["study_id"].astype(int)

        print(f"Dataset size before merging with splits dataset: {len(self.__pandas_full_dataset)}")

        # Merge dataset with splits dataset for faster processing.
        self.__pandas_full_dataset = self.__pandas_full_dataset.merge(
            self.__splits_dataset,
            left_on=["subject_id", "study_id", "dicom_id"],
            right_on=["subject_id", "study_id", "dicom_id"],
            suffixes=("", "_splits")
        )

        print(f"Dataset size after merging with splits dataset: {len(self.__pandas_full_dataset)}")

        # Merge dataset with labels dataset for faster processing.
        self.__pandas_full_dataset = self.__pandas_full_dataset.merge(
            self.__labels_dataset,
            left_on=["subject_id", "study_id"],
            right_on=["subject_id", "study_id"],
            suffixes=("", "_labels")
        )

        print(f"Dataset size after merging with labels dataset: {len(self.__pandas_full_dataset)}")

        # Create labels and replace empty lists with ["No Finding"].
        self.__pandas_full_dataset["labels"] = self.__pandas_full_dataset.apply(lambda row: row.index[row == 1.0].tolist(), axis=1)
        self.__pandas_full_dataset["labels"] = self.__pandas_full_dataset["labels"].apply(lambda x: ["No Finding"] if not x else x)

        if self.__binary_label:
            self.__pandas_full_dataset["labels"] = self.__pandas_full_dataset["labels"].apply(lambda x: [self.__binary_label] if self.__binary_label in x else [])

        # Prepend GCS URI to the file name.
        self.__pandas_full_dataset["file_name"] = self.__pandas_full_dataset["file_name"].apply(lambda x: os.path.join(self.__gcs_uri, x))

        print("Generated dataset:")
        print(self.__pandas_full_dataset.head(50))

    def __create_splits(self):
        print("Creating splits")
        if self.__binary_label:
            # TODO: Parametrize.
            seed = 42
            split_ratio = 0.9

            # Create a dataset of positive samples and a dataset of negative samples.
            pos_df = self.__pandas_full_dataset[self.__pandas_full_dataset["labels"].apply(lambda x: x == [self.__binary_label])]
            neg_df = self.__pandas_full_dataset[self.__pandas_full_dataset["labels"].apply(lambda x: x == [])]
            print(f"Number of positive samples: {len(pos_df)}")
            print(f"Number of negative samples: {len(neg_df)}")

            # Shuffle both datasets.
            pos_df = pos_df.sample(frac=1, random_state=seed).reset_index(drop=True)
            neg_df = neg_df.sample(frac=1, random_state=seed).reset_index(drop=True)
            print(f"Number of positive samples after shuffling: {len(pos_df)}")
            print(f"Number of negative samples after shuffling: {len(neg_df)}")

            # Create split.
            boundary = int(len(pos_df) * split_ratio)
            pos_train = pos_df.iloc[:boundary]
            pos_validation = pos_df.iloc[boundary:]
            neg_train = neg_df.iloc[:boundary]
            neg_validation = neg_df.iloc[boundary:]
            print(f"Number of positive samples in training dataset: {len(pos_train)}")
            print(f"Number of negative samples in training dataset: {len(neg_train)}")
            print(f"Number of positive samples in validation dataset: {len(pos_validation)}")
            print(f"Number of negative samples in validation dataset: {len(neg_validation)}")

            # Merge splits.
            self.__pandas_train_dataset = pd.concat([pos_train, neg_train]).reset_index(drop=True)
            self.__pandas_validation_dataset = pd.concat([pos_validation, neg_validation]).reset_index(drop=True)
            self.__pandas_test_dataset = None

            # Perform shuffling.
            self.__pandas_train_dataset = self.__pandas_train_dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
            self.__pandas_validation_dataset = self.__pandas_validation_dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
        else:
            self.__pandas_train_dataset = self.__pandas_full_dataset.loc[self.__pandas_full_dataset["split"] == "train"]
            self.__pandas_validation_dataset = self.__pandas_full_dataset.loc[self.__pandas_full_dataset["split"] == "validate"]
            self.__pandas_test_dataset = self.__pandas_full_dataset.loc[self.__pandas_full_dataset["split"] == "test"]

        print(f"Train dataset size: {len(self.__pandas_train_dataset)}")
        print(f"Validation dataset size: {len(self.__pandas_validation_dataset)}")
        print(f"Test dataset size: {len(self.__pandas_test_dataset) if self.__pandas_test_dataset else 0}")

        print("")
        print("Train dataset:")
        print(self.__pandas_train_dataset.head(10))

        print("")
        print("Validation dataset:")
        print(self.__pandas_validation_dataset.head(10))

        print("")
        print("Test dataset:")
        print(f"{self.__pandas_test_dataset.head(10) if self.__pandas_test_dataset else None}")

    def __create_torch_datasets(self):
        print("Creating Torch datasets")
        self.__torch_train_dataset = MimicTwoTorchDataset(pandas_dataframe=self.__pandas_train_dataset)
        self.__torch_validation_dataset = MimicTwoTorchDataset(pandas_dataframe=self.__pandas_validation_dataset)
        self.__torch_test_dataset = MimicTwoTorchDataset(pandas_dataframe=self.__pandas_test_dataset) if self.__pandas_test_dataset else None


class MimicTwoTorchDataset(Dataset):
    def __init__(self, pandas_dataframe):
        self.__pandas_dataframe = pandas_dataframe

    def __getitem__(self, idx):
        item = self.__pandas_dataframe.iloc[idx]
        return item

    def __len__(self):
        return len(self.__pandas_dataframe)
