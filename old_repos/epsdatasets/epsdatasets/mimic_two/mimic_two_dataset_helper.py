import os

import datasets
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from epsdatasets.base.base_dataset_helper import BaseDatasetHelper

IMAGE_FILENAMES_FILE = "IMAGE_FILENAMES"
IMAGE_FILENAMES_CORRECTED_FILE = "IMAGE_FILENAMES_CORRECTED"
REPORT_FILENAMES_CORRECTED_FILE = "REPORT_FILENAMES_CORRECTED"
METADATA_FILE = "mimic-cxr-2.0.0-metadata.csv.gz"
SPLIT_FILE = "mimic-cxr-2.0.0-split.csv.gz"
CHEXPERT_FILE = "mimic-cxr-2.0.0-chexpert.csv.gz"
NEGBIO_FILE = "mimic-cxr-2.0.0-negbio.csv.gz"
NO_FINDING_INDEX = 8


class MimicTwoDatasetHelper(BaseDatasetHelper):
    def __init__(self, dataset_path, labels_generator="chexpert", group_multi_images=False):
        self._group_multi_images = group_multi_images
        super().__init__(dataset_path=dataset_path, labels_generator=labels_generator)

    def _load_dataset(self, *args, **kwargs):
        dataset_path = (
            kwargs["dataset_path"]
            if "dataset_path" in kwargs
            else next((arg for arg in args if arg == "dataset_path"), None)
        )
        labels_generator = (
            kwargs["labels_generator"]
            if "labels_generator" in kwargs
            else next((arg for arg in args if arg == "labels_generator"), None)
        )

        print(f"label generator: {labels_generator}")
        if labels_generator == "chexpert":
            labels_file = CHEXPERT_FILE
        elif labels_generator == "negbio":
            labels_file = NEGBIO_FILE
        else:
            raise ValueError(
                "Unsupported labels generator specified. Choose either 'chexpert' or 'negbio'."
            )

        if os.path.exists(os.path.join(dataset_path, REPORT_FILENAMES_CORRECTED_FILE)):
            self._report_files = pd.read_csv(
                os.path.join(dataset_path, REPORT_FILENAMES_CORRECTED_FILE), header=None
            )
        else:
            print("Not finding report files, please use report_paths_generator.py to generate the files")


        if os.path.exists(os.path.join(dataset_path, IMAGE_FILENAMES_CORRECTED_FILE)):
            print(f"Reading image files from file '{IMAGE_FILENAMES_CORRECTED_FILE}'")
            self._image_files = pd.read_csv(
                os.path.join(dataset_path, IMAGE_FILENAMES_CORRECTED_FILE), header=None
            )
        else:
            print(
                f"File '{IMAGE_FILENAMES_CORRECTED_FILE}' not found, scanning dataset path for images"
            )
            self._image_files = pd.DataFrame(self._quick_find_images(dataset_path))
            print(
                f"Scanning complete, saving found images to file '{IMAGE_FILENAMES_CORRECTED_FILE}'"
            )
            self._image_files[0].to_csv(
                os.path.join(dataset_path, IMAGE_FILENAMES_CORRECTED_FILE),
                index=False,
                header=False,
            )

        inconsistent_image_files = pd.read_csv(
            os.path.join(dataset_path, IMAGE_FILENAMES_FILE), header=None
        )
        print(
            f"{len(self._image_files)} image files found (number of image files in inconsistent file '{IMAGE_FILENAMES_FILE}' is {len(inconsistent_image_files)})"
        )

        print("Parsing Mimic Two annotation files")

        self._labels = pd.read_csv(
            os.path.join(dataset_path, labels_file), compression="gzip"
        )
        self._metadata = pd.read_csv(
            os.path.join(dataset_path, METADATA_FILE), compression="gzip"
        )
        self._split = pd.read_csv(
            os.path.join(dataset_path, SPLIT_FILE), compression="gzip"
        )

        self._get_label_names()
        self._generate_full_dataset()
        self._generate_splits()
        self._create_torch_datasets()

    def get_pil_image(self, item):
        image = Image.open(item["image_file"]).convert("RGB")
        return image

    def get_torch_image(self, item, processor):
        image = Image.open(item["image_file"]).convert("RGB")
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
        return self._pandas_full_dataset

    def get_hugging_face_train_dataset(self):
        return datasets.Dataset.from_pandas(self._pandas_train_dataset)

    def get_hugging_face_validation_dataset(self):
        return datasets.Dataset.from_pandas(self._pandas_validation_dataset)

    def get_hugging_face_test_dataset(self):
        return datasets.Dataset.from_pandas(self._pandas_test_dataset)

    def get_torch_train_dataset(self):
        return self._torch_train_dataset

    def get_torch_validation_dataset(self):
        return self._torch_validation_dataset

    def get_torch_test_dataset(self):
        return self._torch_test_dataset

    def get_torch_train_data_loader(self, collate_function, batch_size, num_workers):
        data_loader = DataLoader(
            self._torch_train_dataset,
            collate_fn=collate_function,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        return data_loader

    def get_torch_validation_data_loader(
        self, collate_function, batch_size, num_workers
    ):
        data_loader = DataLoader(
            self._torch_validation_dataset,
            collate_fn=collate_function,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return data_loader

    def get_torch_test_data_loader(self, collate_function, batch_size, num_workers):
        data_loader = DataLoader(
            self._torch_test_dataset,
            collate_fn=collate_function,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return data_loader

    def label_to_names(self, label):
        names = [
            self.ids_to_labels[index] for index, char in enumerate(label) if char == "1"
        ]
        return names

    def _get_label_names(self):
        columns_list = self._labels.columns.values.tolist()
        label_names = [
            column
            for column in columns_list
            if column not in ["subject_id", "study_id"]
        ]
        print(f"Found the following label names: {label_names}")

        self.all_labels_list = label_names
        self.all_labels_dict = {
            label: i for i, label in enumerate(self.all_labels_list)
        }
        self.ids_to_labels = {i: label for i, label in enumerate(self.all_labels_list)}

        assert self.all_labels_list[NO_FINDING_INDEX] == "No Finding"

    def _generate_full_dataset(self):
        print("Generating full dataset")

        num_rows = len(self._image_files)
        self._pandas_full_dataset = pd.DataFrame(
            {
                "dicom_id": np.empty(num_rows, dtype=object),
                "subject_id": np.empty(num_rows, dtype=object),
                "study_id": np.empty(num_rows, dtype=object),
                "view_position": np.empty(num_rows, dtype=object),
                "study_date": np.empty(num_rows, dtype=object),
                "study_time": np.empty(num_rows, dtype=object),
                "image_file": np.empty(num_rows, dtype=object),
                "report": np.empty(num_rows, dtype=object),
                "labels": np.empty(num_rows, dtype=object),
                "split": np.empty(num_rows, dtype=object),
            }
        )

        # Populate 'image_file' column.
        self._pandas_full_dataset["image_file"] = self._image_files[0]
        self._pandas_full_dataset["report"] = self._report_files[0]

        # Populate 'dicom_id' column.
        self._pandas_full_dataset["dicom_id"] = self._pandas_full_dataset[
            "image_file"
        ].apply(lambda image_file: os.path.splitext(os.path.basename(image_file))[0])

        # Populate 'subject_id' and 'study_id' columns.
        dicom_ids_to_search = self._pandas_full_dataset["dicom_id"].values.tolist()
        self._metadata.set_index("dicom_id", inplace=True)
        result = self._metadata.loc[dicom_ids_to_search]
        self._pandas_full_dataset["subject_id"] = result["subject_id"].values
        self._pandas_full_dataset["study_id"] = result["study_id"].values
        self._pandas_full_dataset["view_position"] = result["ViewPosition"].values
        self._pandas_full_dataset["study_date"] = result["StudyDate"].values
        self._pandas_full_dataset["study_time"] = result["StudyTime"].values

        # Make sure there are no Study ID duplicates in the labels dataset.
        duplicates_found = self._labels["study_id"].duplicated().any()
        assert not duplicates_found

        # Find labels corresponding to the given Study IDs.
        try:
            missing_study_ids = []
            study_ids_to_search = self._pandas_full_dataset["study_id"].values.tolist()
            self._labels.set_index("study_id", inplace=True)
            result = self._labels.loc[study_ids_to_search]
        except Exception as e:
            err_msg = str(e)
            start = err_msg.find("[") + 1
            end = err_msg.find("]", start)
            missing_study_ids = err_msg[start:end]
            missing_study_ids = list(map(int, missing_study_ids.split(", ")))

        if missing_study_ids:
            print(
                f"The following study IDs are missing in the labels dataset: {missing_study_ids}, removing them"
            )
            self._pandas_full_dataset = self._pandas_full_dataset[
                ~self._pandas_full_dataset["study_id"].isin(missing_study_ids)
            ]
            study_ids_to_search = self._pandas_full_dataset["study_id"].values.tolist()
            result = self._labels.loc[study_ids_to_search]

        # Remove 'subject_id' column from the result, set all the labels != 1.0 to 0.0 and
        # obtain multi hot encoding by aggregating all the label columns into a single string.
        del result["subject_id"]

        result = result.applymap(lambda x: 0 if x != 1 else 1)
        result = result.astype(str).agg("".join, axis=1)

        # Populate 'labels' column.
        self._pandas_full_dataset["labels"] = result.values

        # Remove ambiguity, replace all 'zeros-only' labels with 'No Finding-only' label.
        zeros_label = "0" * len(self.all_labels_list)
        no_finding_label = (
            zeros_label[:NO_FINDING_INDEX] + "1" + zeros_label[NO_FINDING_INDEX + 1 :]
        )
        print(
            f"Removing ambiguity, replacing all 'zeros-only' labels '{zeros_label}' with 'No Finding-only' label '{no_finding_label}'"
        )
        self._pandas_full_dataset["labels"] = self._pandas_full_dataset[
            "labels"
        ].apply(lambda label: no_finding_label if label == zeros_label else label)

        # Populate 'split' column.
        dicom_ids_to_search = self._pandas_full_dataset["dicom_id"].values.tolist()
        self._split.set_index("dicom_id", inplace=True)
        result = self._split.loc[dicom_ids_to_search]
        self._pandas_full_dataset["split"] = result["split"].values

        print(
            f"Full dataset consisting of {len(self._pandas_full_dataset)} images generated"
        )

    def _generate_splits(self):
        print("Generating splits")

        # Make sure split is either 'train', 'validate' or 'test'.
        unique_names = self._pandas_full_dataset["split"].unique().tolist()
        assert unique_names == ["train", "validate", "test"]

        # Train split.
        self._pandas_train_dataset = self._pandas_full_dataset[
            self._pandas_full_dataset["split"] == "train"
        ]
        unique_names = self._pandas_train_dataset["split"].unique().tolist()
        assert unique_names == ["train"]
        print(f"Generated train split with {len(self._pandas_train_dataset)} samples")

        # Validation split.
        self._pandas_validation_dataset = self._pandas_full_dataset[
            self._pandas_full_dataset["split"] == "validate"
        ]
        unique_names = self._pandas_validation_dataset["split"].unique().tolist()
        assert unique_names == ["validate"]
        print(
            f"Generated validation split with {len(self._pandas_validation_dataset)} samples"
        )

        # Test split.
        self._pandas_test_dataset = self._pandas_full_dataset[
            self._pandas_full_dataset["split"] == "test"
        ]
        unique_names = self._pandas_test_dataset["split"].unique().tolist()
        assert unique_names == ["test"]
        print(f"Generated test split with {len(self._pandas_test_dataset)} samples")

        # Sanity check.
        num_all_samples = (
            len(self._pandas_train_dataset)
            + len(self._pandas_validation_dataset)
            + len(self._pandas_test_dataset)
        )
        assert len(self._pandas_full_dataset) == num_all_samples

        print(f"Total number of samples in all splits combined: {num_all_samples}")

    def _create_torch_datasets(self):
        self._torch_train_dataset = MimicTwoTorchDataset(
            pandas_dataframe=self._pandas_train_dataset,
            group_multi_images=self._group_multi_images
        )
        self._torch_validation_dataset = MimicTwoTorchDataset(
            pandas_dataframe=self._pandas_validation_dataset,
            group_multi_images=self._group_multi_images
        )
        self._torch_test_dataset = MimicTwoTorchDataset(
            pandas_dataframe=self._pandas_test_dataset,
            group_multi_images=self._group_multi_images
        )

    def _quick_find_images(self, directory):
        for entry in os.scandir(directory):
            if entry.is_dir(follow_symlinks=False):
                yield from self._quick_find_images(entry.path)
            elif entry.is_file() and entry.name.lower().endswith(".jpg"):
                yield entry.path


class MimicTwoTorchDataset(Dataset):
    def __init__(self, pandas_dataframe, group_multi_images=False):
        if group_multi_images:
            pandas_dataframe = self.merge_multi_images(pandas_dataframe)
        self._pandas_dataframe = pandas_dataframe

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self._pandas_dataframe.iloc[idx]
            return item
        elif isinstance(idx, slice):
            return MimicTwoTorchDataset(self._pandas_dataframe.iloc[idx])
        else:
            raise TypeError("Invalid argument type.")

    def __len__(self):
        return len(self._pandas_dataframe)


    def merge_multi_images(self, pandas_dataframe):
        group_cols = ['subject_id', 'study_id', 'study_date', 'study_time']
        grouped_df = pandas_dataframe.groupby(group_cols).agg(lambda x: list(x)).reset_index()

        return grouped_df

if __name__ == "__main__":

    path = "/mnt/data/mimic2-jpg/mimic-cxr-jpg-2.1.0.physionet.org/"
    mimic_two_dataset_helper = MimicTwoDatasetHelper(dataset_path=path, labels_generator="chexpert", group_multi_images=True)

    #mimic_two_dataset_helper.view_full_dataset()
    dataset_train = mimic_two_dataset_helper.get_torch_train_dataset()
    dataset_val = mimic_two_dataset_helper.get_torch_validation_dataset()
    dataset_test = mimic_two_dataset_helper.get_torch_test_dataset()

    print(type(dataset_train))
    print(len(dataset_train), len(dataset_test), len(dataset_val))