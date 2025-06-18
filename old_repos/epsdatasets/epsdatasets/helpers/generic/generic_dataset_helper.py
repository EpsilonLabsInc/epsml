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
from epsutils.image import image_utils, image_augmentation
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
                 perform_label_balancing=True,
                 num_data_augmentations=0,
                 convert_images_to_rgb=True,
                 custom_labels=None,
                 seed=42):

        super().__init__(train_file=train_file,
                         validation_file=validation_file,
                         test_file=test_file,
                         base_path_substitutions=base_path_substitutions,
                         body_part=body_part,
                         merge_val_and_test=merge_val_and_test,
                         treat_uncertain_as_positive=treat_uncertain_as_positive,
                         perform_label_balancing=perform_label_balancing,
                         num_data_augmentations=num_data_augmentations,
                         convert_images_to_rgb=convert_images_to_rgb,
                         custom_labels=custom_labels,
                         seed=seed)

    def _load_dataset(self, *args, **kwargs):
        # Store params.
        self.__train_file = kwargs["train_file"] if "train_file" in kwargs else next((arg for arg in args if arg == "train_file"), None)
        self.__validation_file = kwargs["validation_file"] if "validation_file" in kwargs else next((arg for arg in args if arg == "validation_file"), None)
        self.__test_file = kwargs["test_file"] if "test_file" in kwargs else next((arg for arg in args if arg == "test_file"), None)
        self.__base_path_substitutions = kwargs["base_path_substitutions"] if "base_path_substitutions" in kwargs else next((arg for arg in args if arg == "base_path_substitutions"), None)
        self.__body_part = kwargs["body_part"] if "body_part" in kwargs else next((arg for arg in args if arg == "body_part"), None)
        self.__merge_val_and_test = kwargs["merge_val_and_test"] if "merge_val_and_test" in kwargs else next((arg for arg in args if arg == "merge_val_and_test"), None)
        self.__treat_uncertain_as_positive = kwargs["treat_uncertain_as_positive"] if "treat_uncertain_as_positive" in kwargs else next((arg for arg in args if arg == "treat_uncertain_as_positive"), None)
        self.__perform_label_balancing = kwargs["perform_label_balancing"] if "perform_label_balancing" in kwargs else next((arg for arg in args if arg == "perform_label_balancing"), None)
        self.__num_data_augmentations = kwargs["num_data_augmentations"] if "num_data_augmentations" in kwargs else next((arg for arg in args if arg == "num_data_augmentations"), None)
        self.__convert_images_to_rgb = kwargs["convert_images_to_rgb"] if "convert_images_to_rgb" in kwargs else next((arg for arg in args if arg == "convert_images_to_rgb"), None)
        self.__custom_labels = kwargs["custom_labels"] if "custom_labels" in kwargs else next((arg for arg in args if arg == "custom_labels"), None)
        self.__seed = kwargs["seed"] if "seed" in kwargs else next((arg for arg in args if arg == "seed"), None)

        self.__pandas_train_dataset = None
        self.__pandas_validation_dataset = None
        self.__pandas_test_dataset = None
        self.__torch_train_dataset = None
        self.__torch_validation_dataset = None
        self.__torch_test_dataset = None

        self.__uses_single_label = len(self.__custom_labels) == 1

        # Training dataset.

        print("")
        print("----------------")
        print("Training dataset")
        print("----------------")
        print("")

        if aws_s3_utils.is_aws_s3_uri(self.__train_file):
            print(f"Downloading {self.__train_file}")
            aws_s3_data = aws_s3_utils.split_aws_s3_uri(self.__train_file)
            content = aws_s3_utils.download_file_as_bytes(aws_s3_bucket_name=aws_s3_data["aws_s3_bucket_name"], aws_s3_file_name=aws_s3_data["aws_s3_path"])
            content = BytesIO(content)
        else:
            print(f"Loading {self.__train_file}")
            content = self.__train_file

        self.__pandas_train_dataset = self.__filter_dataset(df=pd.read_csv(content, low_memory=False), body_part=self.__body_part)
        self.__generate_training_labels(self.__pandas_train_dataset)

        if self.__uses_single_label:
            print("Generating balancing statistics for the training dataset")
            num_pos, pos_percent, num_neg, neg_percent = self.__balancing_statistics(self.__pandas_train_dataset)
            print(f"There are {num_pos} ({pos_percent:.2f}%) positive and {num_neg} ({neg_percent:.2f}%) negative samples in the training dataset")

        if self.__perform_label_balancing and self.__uses_single_label:
            print("Balancing training dataset")
            self.__pandas_train_dataset = self.__balance_dataset(self.__pandas_train_dataset)
            print(f"After balancing, the training dataset has {len(self.__pandas_train_dataset)} rows")

        # Validation dataset.

        print("")
        print("------------------")
        print("Validation dataset")
        print("------------------")
        print("")

        if aws_s3_utils.is_aws_s3_uri(self.__validation_file):
            print(f"Downloading {self.__validation_file}")
            aws_s3_data = aws_s3_utils.split_aws_s3_uri(self.__validation_file)
            content = aws_s3_utils.download_file_as_bytes(aws_s3_bucket_name=aws_s3_data["aws_s3_bucket_name"], aws_s3_file_name=aws_s3_data["aws_s3_path"])
            content = BytesIO(content)
        else:
            print(f"Loading {self.__validation_file}")
            content = self.__validation_file

        self.__pandas_validation_dataset = self.__filter_dataset(df=pd.read_csv(content, low_memory=False), body_part=self.__body_part)
        self.__generate_training_labels(self.__pandas_validation_dataset)

        if self.__uses_single_label:
            print("Generating balancing statistics for the validation dataset")
            num_pos, pos_percent, num_neg, neg_percent = self.__balancing_statistics(self.__pandas_validation_dataset)
            print(f"There are {num_pos} ({pos_percent:.2f}%) positive and {num_neg} ({neg_percent:.2f}%) negative samples in the validation dataset")

        if self.__perform_label_balancing and self.__uses_single_label:
            print("Balancing validation dataset")
            self.__pandas_validation_dataset = self.__balance_dataset(self.__pandas_validation_dataset)
            print(f"After balancing, the validation dataset has {len(self.__pandas_validation_dataset)} rows")

        # Test dataset.

        print("")
        print("------------")
        print("Test dataset")
        print("------------")
        print("")

        if aws_s3_utils.is_aws_s3_uri(self.__test_file):
            print(f"Downloading {self.__test_file}")
            aws_s3_data = aws_s3_utils.split_aws_s3_uri(self.__test_file)
            content = aws_s3_utils.download_file_as_bytes(aws_s3_bucket_name=aws_s3_data["aws_s3_bucket_name"], aws_s3_file_name=aws_s3_data["aws_s3_path"])
            content = BytesIO(content)
        else:
            print(f"Loading {self.__test_file}")
            content = self.__test_file

        self.__pandas_test_dataset = self.__filter_dataset(df=pd.read_csv(content, low_memory=False), body_part=self.__body_part)
        self.__generate_training_labels(self.__pandas_test_dataset)

        if self.__uses_single_label:
            print("Generating balancing statistics for the test dataset")
            num_pos, pos_percent, num_neg, neg_percent = self.__balancing_statistics(self.__pandas_test_dataset)
            print(f"There are {num_pos} ({pos_percent:.2f}%) positive and {num_neg} ({neg_percent:.2f}%) negative samples in the test dataset")

        if self.__perform_label_balancing and self.__uses_single_label:
            print("Balancing test dataset")
            self.__pandas_test_dataset = self.__balance_dataset(self.__pandas_test_dataset)
            print(f"After balancing, the test dataset has {len(self.__pandas_test_dataset)} rows")

        # Merge validation and test dataset.
        if self.__merge_val_and_test:
            print("Merging validation and test dataset")
            self.__pandas_validation_dataset = pd.concat([self.__pandas_validation_dataset, self.__pandas_test_dataset], axis=0)
            self.__pandas_test_dataset = None
            print(f"After merging validation and test dataset, validation dataset has {len(self.__pandas_validation_dataset)} rows")

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
        return torch.tensor(item["training_labels"])

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

    def __generate_training_labels(self, df):
        print("Generating training labels")
        training_labels = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            labels = labels_utils.parse_structured_labels(ast.literal_eval(row["structured_labels"]), treat_uncertain_as_positive=self.__treat_uncertain_as_positive)
            assert self.__body_part in labels
            training_labels.append(labels_utils.to_multi_hot_encoding(labels[self.__body_part], self.get_labels()))

        assert len(df) == len(training_labels)
        df["training_labels"] = training_labels

    def __balancing_statistics(self, df):
        assert self.__uses_single_label

        num_pos = df["training_labels"].apply(lambda x: x == [1]).sum()
        num_neg = df["training_labels"].apply(lambda x: x == [0]).sum()

        pos_percent = num_pos / (num_pos + num_neg) * 100
        neg_percent = num_neg / (num_pos + num_neg) * 100

        return num_pos, pos_percent, num_neg, neg_percent

    def __balance_dataset(self, df):
        assert self.__uses_single_label

        pos_df = df[df["training_labels"].apply(lambda x: x == [1])]
        pos_df = self.__apply_data_augmentation(df=pos_df, num_data_augmentations=self.__num_data_augmentations)

        neg_df = df[df["training_labels"].apply(lambda x: x == [0])]
        neg_df = self.__apply_data_augmentation(df=neg_df, num_data_augmentations=0)
        neg_df = neg_df.sample(n=len(pos_df), random_state=self.__seed)

        df = pd.concat([pos_df, neg_df]).reset_index(drop=True)
        df = df.sample(frac=1, random_state=self.__seed).reset_index(drop=True)

        return df

    def __apply_data_augmentation(self, df, num_data_augmentations):
        res_df = df.copy(deep=True)
        res_df["augmentation_params"] = None

        if num_data_augmentations > 0:
            print("Applying data augmentation")

            num_augmented_images = len(res_df) * num_data_augmentations
            augmentation_params = image_augmentation.generate_augmentation_parameters(num_images=num_augmented_images, seed=self.__seed)
            augmented_dataset = pd.concat([res_df] * num_data_augmentations, ignore_index=True)
            assert len(augmentation_params) == len(augmented_dataset)
            augmented_dataset["augmentation_params"] = augmentation_params

            org_size = len(res_df)
            res_df = pd.concat([res_df, augmented_dataset], ignore_index=True)

            print(f"Data augmented dataset has {len(res_df)} rows")
            print("Data augmented dataset head:")
            print(res_df.head())

            print("Sanity check")
            for index in range(3):
                for i in range(num_data_augmentations):
                    row = res_df.iloc[index + i * org_size]
                    print(f"{row['image_paths']}: {row['augmentation_params']}")

        return res_df

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

                # If any chest classification element is None, skip the row.
                if any(elem is None for elem in chest_classification):
                    continue

                # All elementes of the chest classification column must be chests!
                if not all(elem.strip().lower() == "chest" for elem in chest_classification):
                    continue

                # If no projection classification was done for this row, skip it.
                if pd.isna(row["projection_classification"]):
                    continue

                projection_classification = ast.literal_eval(row["projection_classification"])
                assert len(projection_classification) == len(image_paths)

                # If any projection classification element is None, skip the row.
                if any(elem is None for elem in projection_classification):
                    continue

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
