import ast
import os
from io import BytesIO

import datasets
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from epsdatasets.helpers.base.base_dataset_helper import BaseDatasetHelper
from epsutils.aws import aws_s3_utils
from epsutils.dicom import dicom_utils
from epsutils.image import image_utils
from epsutils.labels import labels_utils
from epsutils.labels.labels_by_body_part import LABELS_BY_BODY_PART


class GenericDatasetHelper(BaseDatasetHelper):
    def __init__(self,
                 train_file,
                 validation_file,
                 test_file,
                 base_path_substitutions,
                 body_part,
                 merge_val_and_test=True,
                 treat_uncertain_as_positive=True,
                 convert_images_to_rgb=True,
                 custom_labels=None):

        super().__init__(train_file=train_file,
                         validation_file=validation_file,
                         test_file=test_file,
                         base_path_substitutions=base_path_substitutions,
                         body_part=body_part,
                         merge_val_and_test=merge_val_and_test,
                         treat_uncertain_as_positive=treat_uncertain_as_positive,
                         convert_images_to_rgb=convert_images_to_rgb,
                         custom_labels=custom_labels)

    def _load_dataset(self, *args, **kwargs):
        # Store params.
        self.__train_file = kwargs["train_file"] if "train_file" in kwargs else next((arg for arg in args if arg == "train_file"), None)
        self.__validation_file = kwargs["validation_file"] if "validation_file" in kwargs else next((arg for arg in args if arg == "validation_file"), None)
        self.__test_file = kwargs["test_file"] if "test_file" in kwargs else next((arg for arg in args if arg == "test_file"), None)
        self.__base_path_substitutions = kwargs["base_path_substitutions"] if "base_path_substitutions" in kwargs else next((arg for arg in args if arg == "base_path_substitutions"), None)
        self.__body_part = kwargs["body_part"] if "body_part" in kwargs else next((arg for arg in args if arg == "body_part"), None)
        self.__merge_val_and_test = kwargs["merge_val_and_test"] if "merge_val_and_test" in kwargs else next((arg for arg in args if arg == "merge_val_and_test"), None)
        self.__treat_uncertain_as_positive = kwargs["treat_uncertain_as_positive"] if "treat_uncertain_as_positive" in kwargs else next((arg for arg in args if arg == "treat_uncertain_as_positive"), None)
        self.__convert_images_to_rgb = kwargs["convert_images_to_rgb"] if "convert_images_to_rgb" in kwargs else next((arg for arg in args if arg == "convert_images_to_rgb"), None)
        self.__custom_labels = kwargs["custom_labels"] if "custom_labels" in kwargs else next((arg for arg in args if arg == "custom_labels"), None)

        self.__pandas_train_dataset = None
        self.__pandas_validation_dataset = None
        self.__pandas_test_dataset = None
        self.__torch_train_dataset = None
        self.__torch_validation_dataset = None
        self.__torch_test_dataset = None

        # Train dataset.
        if aws_s3_utils.is_aws_s3_uri(self.__train_file):
            print(f"Downloading {self.__train_file}")
            aws_s3_data = aws_s3_utils.split_aws_s3_uri(self.__train_file)
            content = aws_s3_utils.download_file_as_bytes(aws_s3_bucket_name=aws_s3_data["aws_s3_bucket_name"], aws_s3_file_name=aws_s3_data["aws_s3_path"])
            content = BytesIO(content)
        else:
            print(f"Loading {self.__train_file}")
            content = self.__train_file
        self.__pandas_train_dataset = self.__filter_dataset(df=pd.read_csv(content, low_memory=False), body_part=self.__body_part)

        # Validation dataset.
        if aws_s3_utils.is_aws_s3_uri(self.__validation_file):
            print(f"Downloading {self.__validation_file}")
            aws_s3_data = aws_s3_utils.split_aws_s3_uri(self.__validation_file)
            content = aws_s3_utils.download_file_as_bytes(aws_s3_bucket_name=aws_s3_data["aws_s3_bucket_name"], aws_s3_file_name=aws_s3_data["aws_s3_path"])
            content = BytesIO(content)
        else:
            print(f"Loading {self.__validation_file}")
            content = self.__validation_file
        self.__pandas_validation_dataset = self.__filter_dataset(df=pd.read_csv(content, low_memory=False), body_part=self.__body_part)

        # Test dataset.
        if aws_s3_utils.is_aws_s3_uri(self.__test_file):
            print(f"Downloading {self.__test_file}")
            aws_s3_data = aws_s3_utils.split_aws_s3_uri(self.__test_file)
            content = aws_s3_utils.download_file_as_bytes(aws_s3_bucket_name=aws_s3_data["aws_s3_bucket_name"], aws_s3_file_name=aws_s3_data["aws_s3_path"])
            content = BytesIO(content)
        else:
            print(f"Loading {self.__test_file}")
            content = self.__test_file
        self.__pandas_test_dataset = self.__filter_dataset(df=pd.read_csv(content, low_memory=False), body_part=self.__body_part)

        # Merge validation and test dataset.
        if self.__merge_val_and_test:
            self.__pandas_validation_dataset = pd.concat([self.__pandas_validation_dataset, self.__pandas_test_dataset], axis=0)
            self.__pandas_test_dataset = None

        # Create Torch datasets.
        print("Creating Torch datasets")
        self.__torch_train_dataset = GenericTorchDataset(pandas_dataframe=self.__pandas_train_dataset)
        self.__torch_validation_dataset = GenericTorchDataset(pandas_dataframe=self.__pandas_validation_dataset)
        self.__torch_test_dataset = GenericTorchDataset(pandas_dataframe=self.__pandas_test_dataset) if self.__pandas_test_dataset else None

    def get_pil_image(self, item):
        image_paths = item["image_paths"] if isinstance(item["image_paths"], list) else ast.literal_eval(item["image_paths"])
        base_path = item["base_path"]
        images = []

        for image_path in image_paths:
            try:
                subst = self.__base_path_substitutions[base_path]
                image_path = os.path.join(subst, image_path)
                image = dicom_utils.get_dicom_image_fail_safe(image_path, custom_windowing_parameters={"window_center": 0, "window_width": 0})
                image = image_utils.numpy_array_to_pil_image(image, convert_to_rgb=self.__convert_images_to_rgb)
                images.append(image)
            except Exception as e:
                print(f"Error loading {image_path}: {str(e)}")
                raise

        return images

    def get_torch_image(self, item, processor):
        raise NotImplementedError("Not implemented")

    def get_labels(self):
        if self.__custom_labels:
            return self.__custom_labels
        else:
            return LABELS_BY_BODY_PART[self.__body_part]

    def get_ids_to_labels(self):
        raise NotImplementedError("Not implemented")

    def get_labels_to_ids(self):
        raise NotImplementedError("Not implemented")

    def get_torch_label(self, item):
        structured_labels = ast.literal_eval(item["structured_labels"])
        parsed_labels = labels_utils.parse_structured_labels(structured_labels, treat_uncertain_as_positive=self.__treat_uncertain_as_positive)
        assert self.__body_part in parsed_labels
        labels = parsed_labels[self.__body_part]
        multi_hot_labels = labels_utils.to_multi_hot_encoding(labels, self.get_labels())
        return torch.tensor(multi_hot_labels)

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

    def __filter_dataset(self, df, body_part):
        print("Filtering dataset")
        print(f"Original dataset has {len(df)} rows")

        selected_rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Procesing"):
            image_paths = ast.literal_eval(row["image_paths"])
            df_body_parts = {body_part.strip().lower() for body_part in row["body_part"].split(",")}
            body_part = body_part.strip().lower()

            if body_part not in df_body_parts:
                continue

            # For chest perform additonal checks.
            if body_part == "chest":
                # If no chest classification was done for this row, skip it.
                if pd.isna(row["chest_classification"]):
                    continue

                chest_classification = ast.literal_eval(row["chest_classification"])
                assert len(chest_classification) == len(image_paths)

                # All elementes of the chest classification column must be chests!
                if not all(elem.strip().lower() == "chest" for elem in chest_classification):
                    continue

                # If no projection classification was done for this row, skip it.
                if pd.isna(row["projection_classification"]):
                    continue

                projection_classification = ast.literal_eval(row["projection_classification"])
                assert len(projection_classification) == len(image_paths)

                # Study needs to have at least one frontal and one lateral image.
                if "Frontal" not in projection_classification or "Lateral" not in projection_classification:
                    continue

                # Keep only the first frontal and the first lateral image.
                frontal_index = next((i for i, elem in enumerate(projection_classification) if elem == "Frontal"), None)
                lateral_index = next((i for i, elem in enumerate(projection_classification) if elem == "Lateral"), None)
                assert frontal_index is not None and lateral_index is not None
                row["image_paths"] = [image_paths[frontal_index], image_paths[lateral_index]]
                row["projection_classification"] = [projection_classification[frontal_index], projection_classification[lateral_index]]
                row["chest_classification"] = [chest_classification[frontal_index], chest_classification[lateral_index]]

                selected_rows.append(row)

            # For body parts other than chest, unroll images.
            else:
                for image_path in image_paths:
                    new_row = row.copy()
                    new_row["image_paths"] = [image_path]
                    selected_rows.append(new_row)

        df = pd.DataFrame(selected_rows)
        print(f"Filtered dataset has {len(df)} rows")

        return df


class GenericTorchDataset(Dataset):
    def __init__(self, pandas_dataframe):
        self.__pandas_dataframe = pandas_dataframe

    def __getitem__(self, idx):
        item = self.__pandas_dataframe.iloc[idx]
        return item

    def __len__(self):
        return len(self.__pandas_dataframe)
