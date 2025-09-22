import ast
import os
from io import BytesIO

import datasets
import pandas as pd
import torch
from PIL import Image
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
                 sub_body_part=None,
                 merge_val_and_test=True,
                 treat_uncertain_as_positive=True,
                 perform_label_balancing=True,
                 num_data_augmentations=0,
                 compute_num_data_augmentations=False,
                 data_augmentation_target=0,
                 data_augmentation_min=0,
                 unroll_images = False,
                 max_study_images=None,
                 enforce_frontal_and_lateral_view_for_chest=False,
                 convert_images_to_rgb=True,
                 replace_dicom_with_png=False,
                 custom_labels=None,
                 seed=42):

        super().__init__(train_file=train_file,
                         validation_file=validation_file,
                         test_file=test_file,
                         base_path_substitutions=base_path_substitutions,
                         body_part=body_part,
                         sub_body_part=sub_body_part,
                         merge_val_and_test=merge_val_and_test,
                         treat_uncertain_as_positive=treat_uncertain_as_positive,
                         perform_label_balancing=perform_label_balancing,
                         num_data_augmentations=num_data_augmentations,
                         compute_num_data_augmentations=compute_num_data_augmentations,
                         data_augmentation_target=data_augmentation_target,
                         data_augmentation_min=data_augmentation_min,
                         unroll_images=unroll_images,
                         max_study_images=max_study_images,
                         enforce_frontal_and_lateral_view_for_chest=enforce_frontal_and_lateral_view_for_chest,
                         convert_images_to_rgb=convert_images_to_rgb,
                         replace_dicom_with_png=replace_dicom_with_png,
                         custom_labels=custom_labels,
                         seed=seed)

    def _load_dataset(self, *args, **kwargs):
        # Store params.
        self.__train_file = kwargs["train_file"] if "train_file" in kwargs else next((arg for arg in args if arg == "train_file"), None)
        self.__validation_file = kwargs["validation_file"] if "validation_file" in kwargs else next((arg for arg in args if arg == "validation_file"), None)
        self.__test_file = kwargs["test_file"] if "test_file" in kwargs else next((arg for arg in args if arg == "test_file"), None)
        self.__base_path_substitutions = kwargs["base_path_substitutions"] if "base_path_substitutions" in kwargs else next((arg for arg in args if arg == "base_path_substitutions"), None)
        self.__body_part = kwargs["body_part"] if "body_part" in kwargs else next((arg for arg in args if arg == "body_part"), None)
        self.__sub_body_part = kwargs["sub_body_part"] if "sub_body_part" in kwargs else next((arg for arg in args if arg == "sub_body_part"), None)
        self.__merge_val_and_test = kwargs["merge_val_and_test"] if "merge_val_and_test" in kwargs else next((arg for arg in args if arg == "merge_val_and_test"), None)
        self.__treat_uncertain_as_positive = kwargs["treat_uncertain_as_positive"] if "treat_uncertain_as_positive" in kwargs else next((arg for arg in args if arg == "treat_uncertain_as_positive"), None)
        self.__perform_label_balancing = kwargs["perform_label_balancing"] if "perform_label_balancing" in kwargs else next((arg for arg in args if arg == "perform_label_balancing"), None)
        self.__num_data_augmentations = kwargs["num_data_augmentations"] if "num_data_augmentations" in kwargs else next((arg for arg in args if arg == "num_data_augmentations"), None)
        self.__compute_num_data_augmentations = kwargs["compute_num_data_augmentations"] if "compute_num_data_augmentations" in kwargs else next((arg for arg in args if arg == "compute_num_data_augmentations"), None)
        self.__data_augmentation_target = kwargs["data_augmentation_target"] if "data_augmentation_target" in kwargs else next((arg for arg in args if arg == "data_augmentation_target"), None)
        self.__data_augmentation_min = kwargs["data_augmentation_min"] if "data_augmentation_min" in kwargs else next((arg for arg in args if arg == "data_augmentation_min"), None)
        self.__unroll_images = kwargs["unroll_images"] if "unroll_images" in kwargs else next((arg for arg in args if arg == "unroll_images"), None)
        self.__max_study_images = kwargs["max_study_images"] if "max_study_images" in kwargs else next((arg for arg in args if arg == "max_study_images"), None)
        self.__enforce_frontal_and_lateral_view_for_chest = kwargs["enforce_frontal_and_lateral_view_for_chest"] if "enforce_frontal_and_lateral_view_for_chest" in kwargs else next((arg for arg in args if arg == "enforce_frontal_and_lateral_view_for_chest"), None)
        self.__convert_images_to_rgb = kwargs["convert_images_to_rgb"] if "convert_images_to_rgb" in kwargs else next((arg for arg in args if arg == "convert_images_to_rgb"), None)
        self.__replace_dicom_with_png = kwargs["replace_dicom_with_png"] if "replace_dicom_with_png" in kwargs else next((arg for arg in args if arg == "replace_dicom_with_png"), None)
        self.__custom_labels = kwargs["custom_labels"] if "custom_labels" in kwargs else next((arg for arg in args if arg == "custom_labels"), None)
        self.__seed = kwargs["seed"] if "seed" in kwargs else next((arg for arg in args if arg == "seed"), None)

        self.__pandas_train_dataset = None
        self.__pandas_validation_dataset = None
        self.__pandas_test_dataset = None
        self.__torch_train_dataset = None
        self.__torch_validation_dataset = None
        self.__torch_test_dataset = None

        self.__uses_single_label = False if self.__custom_labels is None else len(self.__custom_labels) == 1

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

        self.__pandas_train_dataset = self.__filter_dataset(df=pd.read_csv(content, low_memory=False), body_part=self.__body_part if self.__sub_body_part is None else self.__sub_body_part, use_dicom=self.__sub_body_part is not None)
        self.__generate_training_labels(self.__pandas_train_dataset)

        if self.__uses_single_label:
            print("Generating balancing statistics for the training dataset")
            num_pos, pos_percent, num_neg, neg_percent = self.__balancing_statistics(self.__pandas_train_dataset)
            print(f"There are {num_pos} ({pos_percent:.2f}%) positive and {num_neg} ({neg_percent:.2f}%) negative samples in the training dataset")

        if self.__uses_single_label and self.__compute_num_data_augmentations:
            if num_pos < self.__data_augmentation_min:
                raise ValueError(f"At least {self.__data_augmentation_min} positive training samples required to apply data augmentation")

            if num_pos >= self.__data_augmentation_target:
                self.__num_data_augmentations = 0
                print(f"Data augmenation is not necessary, there are {num_pos} training samples in the training dataset which is enough")
            else:
                self.__num_data_augmentations = self.__data_augmentation_target // num_pos
                print(f"Number of data augmentations computed: {self.__num_data_augmentations}")

        if self.__perform_label_balancing and self.__uses_single_label:
            print("Balancing training dataset")
            self.__pandas_train_dataset = self.__balance_dataset(self.__pandas_train_dataset, apply_data_augmentation=True)
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

        self.__pandas_validation_dataset = self.__filter_dataset(df=pd.read_csv(content, low_memory=False), body_part=self.__body_part if self.__sub_body_part is None else self.__sub_body_part, use_dicom=self.__sub_body_part is not None)
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

        self.__pandas_test_dataset = self.__filter_dataset(df=pd.read_csv(content, low_memory=False), body_part=self.__body_part if self.__sub_body_part is None else self.__sub_body_part, use_dicom=self.__sub_body_part is not None)
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
        self.__torch_test_dataset = GenericTorchDataset(pandas_dataframe=self.__pandas_test_dataset) if self.__pandas_test_dataset is not None else None

    def get_pil_image(self, item):
        image_paths = item["relative_image_paths"] if isinstance(item["relative_image_paths"], list) else ast.literal_eval(item["relative_image_paths"])
        base_path = item["base_path"]
        images = []

        for image_path in image_paths:
            try:
                subst = self.__base_path_substitutions[base_path]
                image_path = os.path.join(subst, image_path)

                if self.__replace_dicom_with_png:
                    image_path = image_path.replace(".dcm", ".png")

                if image_path.endswith(".dcm"):
                    image = dicom_utils.get_dicom_image_fail_safe(image_path, custom_windowing_parameters={"window_center": 0, "window_width": 0})
                    image = image_utils.numpy_array_to_pil_image(image, convert_to_rgb=self.__convert_images_to_rgb)
                else:
                    image = Image.open(image_path)

                if "augmentation_params" in item and item["augmentation_params"] is not None:
                    image = image_augmentation.augment_image(image=image,
                                                             rotation_in_degrees=item["augmentation_params"]["rotation_in_degrees"],
                                                             scaling=item["augmentation_params"]["scaling"],
                                                             translation=item["augmentation_params"]["translation"])

                images.append(image)
            except Exception as e:
                print(f"Error loading {image_path}: {str(e)}")
                raise

        return images

    def get_torch_image(self, item, processor):
        raise NotImplementedError("Not implemented")

    def get_report_text(self, item):
        return item["report_text"].replace("\\n", "\n") if item["report_text"] is not None else None

    def get_labels(self):
        if self.__custom_labels:
            return self.__custom_labels
        else:
            return LABELS_BY_BODY_PART[self.__body_part]

    def get_all_possible_binary_labels_distribution(self):
        org_custom_labels = self.__custom_labels
        org_uses_single_label = self.__uses_single_label

        all_possible_binary_labels = LABELS_BY_BODY_PART[self.__body_part]
        datasets = {
            "Trainining": self.__pandas_train_dataset,
            "Validation": self.__pandas_validation_dataset,
            "Test": self.__pandas_test_dataset
        }
        res = {}

        for dataset_name, df in datasets.items():
            print("")
            print("-----------------------------")
            print(f"{dataset_name} dataset")
            print("-----------------------------")
            print("")

            df_copy = df.copy(deep=True)
            res[dataset_name] = []

            for label in all_possible_binary_labels:
                print(f"Label: {label}")
                self.__custom_labels = [label]
                self.__uses_single_label = True
                self.__generate_training_labels(df_copy)
                num_pos, pos_percent, num_neg, neg_percent = self.__balancing_statistics(df_copy)
                print(f"There are {num_pos} ({pos_percent:.2f}%) positive and {num_neg} ({neg_percent:.2f}%) negative samples in the dataset")
                res[dataset_name].append(f"{label},{num_pos} ({pos_percent:.2f}%),{num_neg} ({neg_percent:.2f}%)")

        self.__custom_labels = org_custom_labels
        self.__uses_single_label = org_uses_single_label

        return res

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
                                 persistent_workers=True,
                                 drop_last=True)
        return data_loader

    def get_torch_validation_data_loader(self, collate_function, batch_size, num_workers):
        data_loader = DataLoader(self.__torch_validation_dataset,
                                 collate_fn=collate_function,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 persistent_workers=True,
                                 drop_last=True)
        return data_loader

    def get_torch_test_data_loader(self, collate_function, batch_size, num_workers):
        data_loader = DataLoader(self.__torch_test_dataset,
                                 collate_fn=collate_function,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 persistent_workers=True,
                                 drop_last=True) if self.__torch_test_dataset else None
        return data_loader

    def __generate_training_labels(self, df):
        print(f"Generating training labels for {self.get_labels()}")
        training_labels = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            labels = labels_utils.parse_structured_labels(ast.literal_eval(row["structured_labels"]), treat_uncertain_as_positive=self.__treat_uncertain_as_positive)

            if self.__body_part not in labels and self.__sub_body_part is None:
                assert self.__body_part in labels
            elif self.__body_part not in labels and self.__sub_body_part is not None:
                labels = []
            else:
                labels = labels[self.__body_part]

            training_labels.append(labels_utils.to_multi_hot_encoding(labels, self.get_labels()))

        assert len(df) == len(training_labels)
        df["training_labels"] = training_labels

    def __balancing_statistics(self, df):
        assert self.__uses_single_label

        num_pos = df["training_labels"].apply(lambda x: x == [1]).sum()
        num_neg = df["training_labels"].apply(lambda x: x == [0]).sum()

        pos_percent = num_pos / (num_pos + num_neg) * 100
        neg_percent = num_neg / (num_pos + num_neg) * 100

        return num_pos, pos_percent, num_neg, neg_percent

    def __balance_dataset(self, df, apply_data_augmentation=False):
        assert self.__uses_single_label

        pos_df = df[df["training_labels"].apply(lambda x: x == [1])]
        if apply_data_augmentation:
            pos_df = self.__apply_data_augmentation(df=pos_df, num_data_augmentations=self.__num_data_augmentations)

        neg_df = df[df["training_labels"].apply(lambda x: x == [0])]
        if apply_data_augmentation:
            neg_df = self.__apply_data_augmentation(df=neg_df, num_data_augmentations=0)
        neg_df = neg_df.sample(n=len(pos_df), random_state=self.__seed, replace=True)

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
                    print(f"{row['relative_image_paths']}: {row['augmentation_params']}")

        return res_df

    def __filter_dataset(self, df, body_part, use_dicom):
        print("Filtering dataset")
        print(f"Original dataset has {len(df)} rows")

        body_part = body_part.strip().lower()
        selected_rows = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Procesing"):
            base_path = row["base_path"]
            if base_path not in self.__base_path_substitutions:
                raise ValueError(f"Base path '{base_path}' not in base path substitutions")
            elif self.__base_path_substitutions[base_path] is None:
                continue

            if use_dicom:
                df_body_parts = row["body_part_dicom"].lower() if pd.notna(row["body_part_dicom"]) else ""
            else:
                df_body_parts = {body_part.strip().lower() for body_part in row["body_part"].split(",")}

            if body_part not in df_body_parts:
                continue

            if pd.isna(row["relative_image_paths"]):
                continue

            image_paths = ast.literal_eval(row["relative_image_paths"])

            # For chest perform additonal checks.
            if body_part == "chest" and self.__enforce_frontal_and_lateral_view_for_chest:
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
                row["relative_image_paths"] = [image_paths[frontal_index], image_paths[lateral_index]]
                row["projection_classification"] = [projection_classification[frontal_index], projection_classification[lateral_index]]
                row["chest_classification"] = [chest_classification[frontal_index], chest_classification[lateral_index]]

                selected_rows.append(row)

            # For body parts other than chest, either unroll images if unrolling is enabled...
            elif self.__unroll_images:
                if self.__max_study_images is not None and len(image_paths) > self.__max_study_images:
                    continue

                for image_path in image_paths:
                    new_row = row.copy()
                    new_row["relative_image_paths"] = [image_path]
                    selected_rows.append(new_row)

            # ... or keep all study images as they are.
            else:
                if self.__max_study_images is not None and len(image_paths) > self.__max_study_images:
                    continue
                elif len(image_paths) == 0:
                    continue

                selected_rows.append(row)

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


if __name__ == "__main__":
    TRAIN_FILE = "/mnt/efs/all-cxr/combined/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_train.csv"
    VALIDATION_FILE = "/mnt/efs/all-cxr/combined/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_val.csv"
    TEST_FILE = "/mnt/efs/all-cxr/combined/gradient_batches_1-5_segmed_batches_1-4_simonmed_batches_1-10_reports_with_labels_test.csv"
    BODY_PART = "Chest"
    TREAT_UNCERTAIN_AS_POSITIVE = True

    helper = GenericDatasetHelper(train_file=TRAIN_FILE,
                                  validation_file=VALIDATION_FILE,
                                  test_file=TEST_FILE,
                                  base_path_substitutions=None,
                                  body_part=BODY_PART,
                                  merge_val_and_test=False,
                                  treat_uncertain_as_positive=TREAT_UNCERTAIN_AS_POSITIVE,
                                  perform_label_balancing=False)

    res = helper.get_all_possible_binary_labels_distribution()

    print("")
    print(f"Distribution of all possible binary labels for body part {BODY_PART}:")
    print("")

    for dataset_name, vals in res.items():
        print(f"{dataset_name} dataset:")
        for val in vals:
            print(val)
        print("")
