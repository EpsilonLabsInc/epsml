import io
import os

import datasets
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from epsdatasets.base.base_dataset_helper import BaseDatasetHelper


class MimicOneDatasetHelper(BaseDatasetHelper):
    def __init__(self,
                 dataset_url,
                 download_dir,
                 load_from_file_if_exists=False,
                 train_test_split=0.9,
                 seed=None,
                 perform_healing=False):

        super().__init__(dataset_url=dataset_url,
                         download_dir=download_dir,
                         load_from_file_if_exists=load_from_file_if_exists,
                         train_test_split=train_test_split,
                         seed=seed,
                         perform_healing=perform_healing)

    def _load_dataset(self, *args, **kwargs):
        dataset_url = kwargs["dataset_url"] if "dataset_url" in kwargs else next((arg for arg in args if arg == "dataset_url"), None)
        download_dir = kwargs["download_dir"] if "download_dir" in kwargs else next((arg for arg in args if arg == "download_dir"), None)
        load_from_file_if_exists = kwargs["load_from_file_if_exists"] if "load_from_file_if_exists" in kwargs else next((arg for arg in args if arg == "load_from_file_if_exists"), None)
        train_test_split = kwargs["train_test_split"] if "train_test_split" in kwargs else next((arg for arg in args if arg == "train_test_split"), None)
        seed = kwargs["seed"] if "seed" in kwargs else next((arg for arg in args if arg == "seed"), None)
        perform_healing = kwargs["perform_healing"] if "perform_healing" in kwargs else next((arg for arg in args if arg == "perform_healing"), None)

        assert(0.0 <= train_test_split <= 1.0)

        # Download the dataset (or load from file if it exists).
        self.__download(dataset_url, download_dir, load_from_file_if_exists)

        # Remove invalid samples from the dataset.
        if perform_healing:
            self.__heal_dataset()

        # Obtain all the labels.
        self.__get_all_labels()

        # Split dataset into train and test.
        self.__split_dataset(train_test_split, seed)

        # Create Torch datasets.
        self.__create_torch_datasets()

        # Run a sanity check.
        self.__run_sanity_check()

    def get_pil_image(self, item):
        image = Image.open(io.BytesIO(item["image"]["bytes"])).convert("RGB")
        return image

    def get_torch_image(self, item, processor):
        image = Image.open(io.BytesIO(item["image"]["bytes"])).convert("RGB")
        return processor(image, return_tensors="pt").pixel_values

    def get_labels(self):
        raise NotImplementedError("Function not implemented")

    def get_ids_to_labels(self):
        raise NotImplementedError("Function not implemented")

    def get_labels_to_ids(self):
        raise NotImplementedError("Function not implemented")

    def get_torch_label(self, item):
        raise NotImplementedError("Function not implemented")

    def get_pandas_full_dataset(self):
        return self.__hugging_face_full_dataset.to_pandas()

    def get_hugging_face_train_dataset(self):
        return self.__hugging_face_train_dataset

    def get_hugging_face_validation_dataset(self):
        raise NotImplementedError("Mimic One dataset does not provide validation dataset")

    def get_hugging_face_test_dataset(self):
        return self.__hugging_face_test_dataset

    def get_torch_train_dataset(self):
        return self.__torch_train_dataset

    def get_torch_validation_dataset(self):
        raise NotImplementedError("Mimic One dataset does not provide validation dataset")

    def get_torch_test_dataset(self):
        return self.__torch_test_dataset

    def get_torch_train_data_loader(self, collate_function, batch_size, num_workers):
        data_loader = DataLoader(self.__torch_train_dataset, collate_fn=collate_function,
                                 batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return data_loader

    def get_torch_validation_data_loader(self, collate_function, batch_size, num_workers):
        raise NotImplementedError("Mimic One dataset does not provide validation dataset")

    def get_torch_test_data_loader(self, collate_function, batch_size, num_workers):
        data_loader = DataLoader(self.__torch_test_dataset, collate_fn=collate_function,
                                 batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return data_loader

    @staticmethod
    def split_into_labels(text):
        labels = text.split(";")
        labels = [label.strip().replace("'", "") for label in labels]
        return labels

    @staticmethod
    def multi_hot_encoding(text, all_labels_list):
        labels = MimicOneDatasetHelper.split_into_labels(text)
        vec = [0] * len(all_labels_list)

        for label in labels:
            if label in all_labels_list:
                index = all_labels_list.index(label)
                vec[index] = 1

        encoded_list = np.array(vec)
        encoded_string = "".join(map(str, encoded_list))
        return encoded_list, encoded_string

    @staticmethod
    def multi_hot_decoding(encoded_string, all_labels_list):
        int_list = list(map(int, encoded_string))
        indices = [index for index, val in enumerate(int_list) if val != 0]
        decoded_list = [all_labels_list[index] for index in indices]
        decoded_string = "; ".join(decoded_list)
        return decoded_list, decoded_string

    def __download(self, dataset_url, download_dir, load_from_file_if_exists):
        dataset_name = os.path.split(dataset_url)[-1]
        dataset_path = os.path.join(download_dir, dataset_name)

        if load_from_file_if_exists and os.path.exists(dataset_path):
            print("Dataset found on disk, loading from there")
            dataset = datasets.load_from_disk(dataset_path)
        else:
            print("Downloading dataset from HuggingFace")
            dataset = datasets.load_dataset(dataset_url)
            dataset.save_to_disk(dataset_path)

        # Mimic One dataset has only a "train" split.
        self.__hugging_face_full_dataset = dataset["train"]

    def __heal_dataset(self):
        def callback(item, index):
            labels = self.split_into_labels(item["text"])
            labels = [label for label in labels if label.lower() not in ["", "chest x-ray", "support devices"]]
            item["text"] = "; ".join(labels)

            if len(item["text"]) > 0:
                valid_indices.append(index)

            return item

        print("Healing dataset")
        print(f"Dataset size BEFORE healing: {len(self.__hugging_face_full_dataset)}")

        valid_indices = []
        dataset = self.__hugging_face_full_dataset.map(callback, with_indices=True, load_from_cache_file=False)
        self.__hugging_face_full_dataset = dataset.select(valid_indices)

        print("Dataset healed")
        print(f"Dataset size AFTER healing: {len(self.__hugging_face_full_dataset)}")

    def __get_all_labels(self):
        def callback(item):
            labels = self.split_into_labels(item["text"])

            for label in labels:
                if label not in self.all_labels_list:
                    self.all_labels_list.append(label)

        print("Obtaining all the labels in the dataset")

        self.all_labels_list = []
        self.__hugging_face_full_dataset.map(callback)
        self.all_labels_dict = {label: i for i, label in enumerate(self.all_labels_list)}
        self.ids_to_labels = {i: label for i, label in enumerate(self.all_labels_list)}

        print(f"All labels: {self.all_labels_list}")
        print(f"All labels dictionary: {self.all_labels_dict}")
        print(f"Ids-to-labels: {self.ids_to_labels}")

    def __split_dataset(self, train_test_split, seed):
        test_size = 1.0 - train_test_split
        print(f"Splitting the dataset, using {test_size} for the test set")

        split_dataset = self.__hugging_face_full_dataset.train_test_split(test_size=test_size, seed=seed)
        self.__hugging_face_train_dataset = split_dataset["train"]
        self.__hugging_face_test_dataset = split_dataset["test"]

        print(f"Size of the train set: {len(self.__hugging_face_train_dataset)}")
        print(f"Size of the test set: {len(self.__hugging_face_test_dataset)}")

    def __create_torch_datasets(self):
        self.__torch_train_dataset = MimicOneTorchDataset(pandas_dataframe=self.__hugging_face_train_dataset.to_pandas())
        self.__torch_test_dataset = MimicOneTorchDataset(pandas_dataframe=self.__hugging_face_test_dataset.to_pandas())

    def __run_sanity_check(self):
        print("Running sanity check")

        for i in range(10):
            item = self.__hugging_face_full_dataset[i]
            print(f"Item {i}: {item['text']}")
            enc_list, enc_string = MimicOneDatasetHelper.multi_hot_encoding(item["text"], self.all_labels_list)
            print(f"Encoded: {enc_string}")
            dec_list, dec_string = MimicOneDatasetHelper.multi_hot_decoding(enc_string, self.all_labels_list)
            print(f"Decoded: {dec_string}")


class MimicOneTorchDataset(Dataset):
    def __init__(self, pandas_dataframe):
        self.__pandas_dataframe = pandas_dataframe

    def __getitem__(self, idx):
        item = self.__pandas_dataframe.iloc[idx]
        return item

    def __len__(self):
        return len(self.__pandas_dataframe)
