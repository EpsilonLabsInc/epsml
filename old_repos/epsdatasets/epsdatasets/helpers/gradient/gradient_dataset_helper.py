import ast
import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
import torch
from google.cloud import storage
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from epsdatasets.helpers.base.base_dataset_helper import BaseDatasetHelper
from epsutils.dicom import dicom_utils
from epsutils.gcs import gcs_utils
from epsutils.labels.grouped_labels_manager import GroupedLabelsManager
from epsutils.math import math_utils


DICOM_TAGS_TO_READ = [
    {"name": "SOPClassUID",               "required": True},
    {"name": "StudyInstanceUID",          "required": True},
    {"name": "SeriesInstanceUID",         "required": True},
    {"name": "InstitutionName",           "required": True},
    {"name": "ImageType",                 "required": True},
    {"name": "InstanceNumber",            "required": True},
    {"name": "PatientID",                 "required": True},
    {"name": "PatientBirthDate",          "required": True},
    {"name": "StudyDate",                 "required": True},
    {"name": "Modality",                  "required": True},
    {"name": "PhotometricInterpretation", "required": True},
    {"name": "Rows",                      "required": True},
    {"name": "Columns",                   "required": True},
    {"name": "SamplesPerPixel",           "required": True},
]


DICOM_MODALITY_MAPPING = {
    pydicom.uid.CTImageStorage: "CT"
}


VALID_INSTITUTION_NAMES = ["tachyeres", "thryothor"]


class GradientDatasetHelper(BaseDatasetHelper):
    def __init__(
            self,
            display_name,
            images_dir,
            reports_file,
            grouped_labels_file,
            images_index_file=None,
            generated_data_file=None,
            output_dir=".",
            perform_quality_check=False,
            gcs_bucket_name=None,
            modality="CT",
            min_volume_depth=None,
            max_volume_depth=None,
            preserve_image_format=False,
            use_half_precision=False,
            seed=None,
            max_num_workers=100,
            run_statistics=False):
        super().__init__(
            display_name=display_name,
            images_dir=images_dir,
            reports_file=reports_file,
            grouped_labels_file=grouped_labels_file,
            images_index_file=images_index_file,
            generated_data_file=generated_data_file,
            output_dir=output_dir,
            perform_quality_check=perform_quality_check,
            gcs_bucket_name=gcs_bucket_name,
            modality=modality,
            min_volume_depth=min_volume_depth,
            max_volume_depth=max_volume_depth,
            preserve_image_format=preserve_image_format,
            use_half_precision=use_half_precision,
            seed=seed,
            max_num_workers=max_num_workers,
            run_statistics=run_statistics)

    def _load_dataset(self, *args, **kwargs):
        self.__display_name = kwargs["display_name"] if "display_name" in kwargs else next((arg for arg in args if arg == "display_name"), None)
        self.__images_dir = kwargs["images_dir"] if "images_dir" in kwargs else next((arg for arg in args if arg == "images_dir"), None)
        self.__reports_file = kwargs["reports_file"] if "reports_file" in kwargs else next((arg for arg in args if arg == "reports_file"), None)
        self.__grouped_labels_file = kwargs["grouped_labels_file"] if "grouped_labels_file" in kwargs else next((arg for arg in args if arg == "grouped_labels_file"), None)
        self.__images_index_file = kwargs["images_index_file"] if "images_index_file" in kwargs else next((arg for arg in args if arg == "images_index_file"), None)
        self.__generated_data_file = kwargs["generated_data_file"] if "generated_data_file" in kwargs else next((arg for arg in args if arg == "generated_data_file"), None)
        self.__output_dir = kwargs["output_dir"] if "output_dir" in kwargs else next((arg for arg in args if arg == "output_dir"), None)
        self.__perform_quality_check = kwargs["perform_quality_check"] if "perform_quality_check" in kwargs else next((arg for arg in args if arg == "perform_quality_check"), None)
        self.__gcs_bucket_name = kwargs["gcs_bucket_name"] if "gcs_bucket_name" in kwargs else next((arg for arg in args if arg == "gcs_bucket_name"), None)
        self.__modality = kwargs["modality"] if "modality" in kwargs else next((arg for arg in args if arg == "modality"), None)
        self.__min_volume_depth = kwargs["min_volume_depth"] if "min_volume_depth" in kwargs else next((arg for arg in args if arg == "min_volume_depth"), None)
        self.__max_volume_depth = kwargs["max_volume_depth"] if "max_volume_depth" in kwargs else next((arg for arg in args if arg == "max_volume_depth"), None)
        self.__preserve_image_format = kwargs["preserve_image_format"] if "preserve_image_format" in kwargs else next((arg for arg in args if arg == "preserve_image_format"), None)
        self.__use_half_precision = kwargs["use_half_precision"] if "use_half_precision" in kwargs else next((arg for arg in args if arg == "use_half_precision"), None)
        self.__seed = kwargs["seed"] if "seed" in kwargs else next((arg for arg in args if arg == "seed"), None)
        self.__max_num_workers = kwargs["max_num_workers"] if "max_num_workers" in kwargs else next((arg for arg in args if arg == "max_num_workers"), None)
        self.__run_statistics = kwargs["run_statistics"] if "run_statistics" in kwargs else next((arg for arg in args if arg == "run_statistics"), None)

        self.__images_index = None
        self.__depth_histogram = {}
        self.__institutions = {}

        self.__reports_dir = os.path.dirname(self.__reports_file) if self.__reports_file is not None else None
        self.__use_gcs = self.__gcs_bucket_name is not None
        self.__modality = self.__modality.upper()
        self.__dtype = torch.float16 if self.__use_half_precision else torch.float32
        self.__custom_windowing_parameters = {"window_center": 0, "window_width": 0}
        self.__use_nifti_files = False

        print(f"Display name: {self.__display_name}")
        print(f"Images directory: {self.__images_dir}")
        print(f"Reports file: {self.__reports_file}")
        print(f"Reports directory: {self.__reports_dir}")
        print(f"Grouped labels file: {self.__grouped_labels_file}")
        print(f"Generated data file: {self.__generated_data_file}")
        print(f"GCS bucket name: {self.__gcs_bucket_name}")

        if self.__perform_quality_check:
            print("Images quality check will be performed")
        else:
            print("Images quality check will be skipped")

        if self.__run_statistics:
            print("Statistics will be saved")
        else:
            print("Statistics won't be saved")

        os.makedirs(self.__output_dir, exist_ok=True)

        if self.__grouped_labels_file is not None:
            print("Creating grouped labels manager")
            self.__create_grouped_labels_manager()
        else:
            print("Grouped labels file not provided, skipping grouped labels manager creation")
            self.__grouped_labels_manager = None

        if self.__generated_data_file is None:
            print("Generating full dataset")
            self.__generate_full_dataset()
        else:
            print(f"Loading generated data from '{self.__generated_data_file}'")
            self.__load_generated_data()

        print("Splitting data")
        self.__split_data()

        print("Creating torch datasets")
        self.__create_torch_datasets()

    def get_max_depth(self):
        raise NotImplementedError("Method not implemented")

    def get_pil_image(self, item, normalization_depth=None, sample_slices=False):

        # Load volume from a single NIfTI file.
        if self.__use_nifti_files:
            nifti_file = item["volume"]["nifti_file"]

            if self.__use_gcs:
                file_dir = os.path.dirname(nifti_file)
                images_dir = self.__images_dir if file_dir == "" else file_dir
                nifti_file = nifti_file if file_dir == "" else os.path.basename(nifti_file)

                gcs_utils.download_file(gcs_bucket_name=self.__gcs_bucket_name,
                                        gcs_file_name=os.path.join(images_dir, nifti_file),
                                        local_file_name=nifti_file,
                                        num_retries=None,  # Retry indefinitely.
                                        show_warning_on_retry=True)

                if not os.path.exists(nifti_file):
                    raise ValueError(f"NIfTI file '{nifti_file}' not properly downloaded")

                sitk_image = sitk.ReadImage(nifti_file)

                if os.path.exists(nifti_file):
                    os.remove(nifti_file)
            else:
                sitk_image = sitk.ReadImage(os.path.join(self.__images_dir, nifti_file))

            sitk_image_array = sitk.GetArrayFromImage(sitk_image)

            # Downsample volume.
            if normalization_depth is not None and sitk_image_array.shape[0] != normalization_depth:
                if sample_slices:
                    if normalization_depth > sitk_image_array.shape[0]:
                        padding = (normalization_depth - sitk_image_array.shape[0]) // 2
                        indices = list(range(sitk_image_array.shape[0]))
                        sitk_image_array = sitk_image_array[indices, :, :]
                        padded_array = np.zeros((normalization_depth, sitk_image_array.shape[1], sitk_image_array.shape[2]), dtype=sitk_image_array.dtype)
                        padded_array[padding:padding+len(indices)] = sitk_image_array
                        sitk_image_array = padded_array
                    else:
                        indices = np.random.choice(sitk_image_array.shape[0], normalization_depth, replace=False)
                        indices = np.sort(indices)
                        sitk_image_array = sitk_image_array[indices, :, :]
                else:
                    new_shape = (normalization_depth, sitk_image_array.shape[1], sitk_image_array.shape[2])
                    sitk_image_array = math_utils.interpolate_volume(input_volume=sitk_image_array, new_shape=new_shape)

            images = [sitk_image_array[i, :, :] for i in range(sitk_image_array.shape[0])]

        # Load volume from multiple DICOM files.
        else:
            images = []
            path = item["volume"]["path"]
            dicom_files = item["volume"]["dicom_files"]

            if len(dicom_files) == 0:
                raise ValueError("No images in the volume")

            if self.__use_gcs:
                client = storage.Client()
                bucket = client.bucket(self.__gcs_bucket_name)

            for dicom_file in dicom_files:
                if self.__use_gcs:
                    blob = bucket.blob(os.path.join(self.__images_dir, path, dicom_file))
                    content = BytesIO(blob.download_as_bytes())
                    image = dicom_utils.get_dicom_image(content, custom_windowing_parameters=self.__custom_windowing_parameters)
                else:
                    image = dicom_utils.get_dicom_image(
                        os.path.join(self.__images_dir, path, dicom_file), custom_windowing_parameters=self.__custom_windowing_parameters)

                images.append(image)

        # Process images.
        for i in range(len(images)):
            if images[i].dtype != np.uint16:
                raise ValueError(f"Image type should be uint16 but got {images[i].dtype} instead")

            if self.__preserve_image_format:
                images[i] = Image.fromarray(images[i], mode="I;16")
            else:
                max = np.max(images[i])
                min = np.min(images[i])
                if max > min:
                    images[i] = ((images[i] - min) / (max - min) * 255).astype(np.uint8)
                else:
                    images[i].fill(0)
                    images[i] = images[i].astype(np.uint8)
                images[i] = Image.fromarray(images[i]).convert("RGB")

        return images

    def get_torch_image(self, item, transform, normalization_depth=None, sample_slices=False):
        images = self.get_pil_image(item, normalization_depth, sample_slices)

        if self.__preserve_image_format:
            tensors = [transform(image).to(self.__dtype) / 65535.0 for image in images]
        else:
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
        return torch.tensor(item["label"], dtype=self.__dtype)

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
                                 batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                 persistent_workers=True)
        return data_loader

    def get_torch_validation_data_loader(self, collate_function, batch_size, num_workers):
        data_loader = DataLoader(self.__torch_validation_dataset, collate_fn=collate_function,
                                 batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 persistent_workers=True)
        return data_loader

    def get_torch_test_data_loader(self, collate_function, batch_size, num_workers):
        data_loader = DataLoader(self.__torch_test_dataset, collate_fn=collate_function,
                                 batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 persistent_workers=True)
        return data_loader

    def __create_grouped_labels_manager(self):
        with open(self.__grouped_labels_file, "r") as json_file:
            grouped_labels = json.load(json_file)

        self.__grouped_labels_manager = GroupedLabelsManager(grouped_labels)  # TODO: Set allow_duplicate_labels to False.

    def __generate_full_dataset(self):
        # Log file.
        log_file_name = os.path.join(self.__output_dir, self.__display_name + "-log.txt")
        logging.basicConfig(
            filename=log_file_name,
            filemode="w",
            format="%(levelname)s: %(message)s",
            datefmt="%y-%m-%d %H:%M:%S",
            level=logging.WARNING)

        # Create images index.
        if not self.__use_gcs:
            if self.__images_index_file is None:
                print("Creating images index (this may take a while)")
                self.__create_images_index()

                print("Saving images index (this may take a while)")
                out_file = os.path.join(self.__output_dir, self.__display_name + "-images_index.json")
                with open(out_file, "w") as json_file:
                    json.dump(self.__images_index, json_file)
            else:
                print("Loading images index")
                with open(self.__images_index_file, "r") as json_file:
                    self.__images_index = json.load(json_file)

        # Reports file is a comma-separated CSV file with an extra header row.
        print("Reading reports file")
        df = pd.read_csv(self.__reports_file, sep=",", low_memory=False)
        num_rows = len(df)

        # Check if grouped labels are included.
        if "grouped_labels" in df:
            use_dummy_labels = False
        else:
            print("Reports file does not have 'grouped_labels' field, dummy labels will be used")
            use_dummy_labels = True

        self.__pandas_full_dataset = pd.DataFrame(columns=["volume", "label"])
        self.__max_depth = 0

        num_all_volumes = 0
        num_invalid_institutions = 0
        num_volumes_above_max = 0
        num_volumes_below_min = 0
        num_valid_slices = 0
        num_valid_volumes = 0
        sum_dt = 0
        avg_dt = 0

        start_time = time.time()

        print("Validating volumes and populating dataset")

        for index, row in df.iterrows():
            if index % 100 == 0:
                # Print progress (clear the entire line and write updated progress in the same line).
                progress = (index + 1) / num_rows * 100.0
                elapsed = time.time() - start_time
                print("\033[2K", end="")
                print(f"Progress: {index + 1}/{num_rows} [{progress:.2f}%]   Elapsed: {elapsed:.2f} sec   Avg processing time per slice: {avg_dt:.4f} sec   "
                    f"Invalid institutions: {num_invalid_institutions}   All volumes: {num_all_volumes}   Volumes below min: {num_volumes_below_min}   "
                    f"Volumes above max: {num_volumes_above_max}   Valid volumes: {num_valid_volumes}   Valid slices: {num_valid_slices}", end="\r")

            institution_name = row["InstitutionName"].lower()
            if not any(name in institution_name for name in VALID_INSTITUTION_NAMES):
                if self.__run_statistics:
                    self.__register_institution(institution_name)

                num_invalid_institutions += 1
                logging.warning(f"Invalid institution '{row['InstitutionName']}' for study at row '{row['row_id']}'")
                continue

            row["SeriesInstanceUid"] = ast.literal_eval(row["SeriesInstanceUid"])
            row["Modality"] = ast.literal_eval(row["Modality"])[0]
            row["PatientBirthDate"] = str(int(row["PatientBirthDate"])) if not math.isnan(row["PatientBirthDate"]) else ""
            row["StudyDate"] = str(int(row["StudyDate"])) if not math.isnan(row["StudyDate"]) else ""

            for series_instance_uid in row["SeriesInstanceUid"]:
                num_all_volumes += 1
                t0 = time.time()

                volume_dir = os.path.join(self.__images_dir, row["PatientID"], row["AccessionNumber"], "studies", row["StudyInstanceUid"], "series", series_instance_uid, "instances/")

                # Get all the DICOM files in the volume dir.
                dicom_files = self.__get_dicom_files_from_gcs(volume_dir) if self.__use_gcs else self.__get_dicom_files_from_disk(volume_dir)

                if dicom_files is None:
                    logging.warning(f"Error getting DICOM files for volume with Series Instance UID '{series_instance_uid}' at row '{row['row_id']}'")
                    continue

                num_dicom_files = len(dicom_files)

                if num_dicom_files == 0:
                    logging.warning(f"No DICOM files found for volume with Series Instance UID '{series_instance_uid}' at row '{row['row_id']}'")
                    continue

                # Filter out volumes below min volume depth.
                if self.__min_volume_depth is not None and num_dicom_files < self.__min_volume_depth:
                    num_volumes_below_min += 1
                    logging.warning(f"Number of slices {num_dicom_files} < {self.__min_volume_depth} "
                                    f"for volume with Series Instance UID '{series_instance_uid}' at row '{row['row_id']}'")
                    continue

                # Filter out volumes above max volume depth.
                if self.__max_volume_depth is not None and num_dicom_files > self.__max_volume_depth:
                    num_volumes_above_max += 1
                    logging.warning(f"Number of slices {num_dicom_files} > {self.__max_volume_depth} "
                                    f"for volume with Series Instance UID '{series_instance_uid}' at row '{row['row_id']}'")
                    continue

                # Read the DICOM files.
                volume = self.__read_dicom_files_from_gcs(dicom_files) if self.__use_gcs else self.__read_dicom_files_from_disk(dicom_files)
                if volume == None:
                    logging.warning(f"Error reading volume with Series Instance UID '{series_instance_uid}' at row '{row['row_id']}'")
                    continue

                # Validate DICOM data and compare it with the report data.
                report_data = row.copy()
                report_data["series_instance_uid"] = series_instance_uid
                res, err_msg = self.__validate_volume(volume=volume, report_data=report_data)
                if not res:
                    logging.warning(f"{err_msg} for volume with Series Instance UID '{series_instance_uid}' at row '{row['row_id']}'")
                    continue

                if self.__run_statistics:
                    self.__register_volume_depth(num_dicom_files)

                # Check volume images quality.
                if self.__perform_quality_check:
                    res, err_msg = dicom_utils.check_dicom_volume_images_quality(volume)
                    if not res:
                        logging.warning(f"Images quality check failed: {err_msg}, skipping volume with Series Instance UID '{series_instance_uid}' at row '{row['row_id']}'")
                        continue

                # Sort the volume.
                volume = self.__sort_volume(volume)

                # Make sure the slices are in ascending and incrementing order.
                if not self.__is_in_ascending_and_incrementing_order(volume):
                    logging.warning(f"Slices are not in ascending and incrementing order "
                                    f"for volume with Series Instance UID '{series_instance_uid}' at row '{row['row_id']}'")
                    continue

                # Get label.
                if use_dummy_labels:
                    label = 0
                else:
                    val = ast.literal_eval(row["grouped_labels"])
                    label = self.__grouped_labels_manager.grouped_sample_labels_to_encoded_list(val)

                # Volume has max depth?
                if len(volume) > self.__max_depth:
                    self.__max_depth = len(volume)

                # Compute average processing time per slice.
                dt = (time.time() - t0) / num_dicom_files
                sum_dt += dt
                num_valid_slices += num_dicom_files
                num_valid_volumes += 1
                avg_dt = sum_dt / num_valid_volumes

                # Reformat volume so that it occupies less memory.
                volume = {
                    "path": os.path.dirname(volume[0]["dicom_file"]),
                    "dicom_files": [os.path.basename(slice["dicom_file"]) for slice in volume]
                }

                # Add to dataset.
                self.__pandas_full_dataset.loc[len(self.__pandas_full_dataset)] = [volume, label]

        print("")
        print(f"Full dataset size: {len(self.__pandas_full_dataset)}")

        # Save statistics.
        if self.__run_statistics:
            print("Saving statistics")
            self.__save_statistics()

        # Save generated data.
        out_file = os.path.join(self.__output_dir, self.__display_name + "-generated_data.csv")
        print(f"Saving generated data to '{out_file}'")
        self.__pandas_full_dataset.to_csv(out_file, index=False)

    def __register_institution(self, institution):
        if institution not in self.__institutions:
            self.__institutions[institution] = 1
        else:
            self.__institutions[institution] += 1

    def __register_volume_depth(self, depth):
        if depth not in self.__depth_histogram:
            self.__depth_histogram[depth] = 1
        else:
            self.__depth_histogram[depth] += 1

    def __save_statistics(self):
        # Save institutions.
        out_file = os.path.join(self.__output_dir, self.__display_name + "-institutions.json")
        with open(out_file, "w") as json_file:
            json.dump(self.__institutions, json_file, indent=4)

        # Sort depth histogram and fill missing depths with zero.
        min_depth = min(self.__depth_histogram.keys())
        max_depth = max(self.__depth_histogram.keys())
        self.__depth_histogram = {k: self.__depth_histogram.get(k, 0) for k in range(min_depth, max_depth + 1)}

        # Save depth histogram as json.
        out_file = os.path.join(self.__output_dir, self.__display_name + "-full_depth_histogram.json")
        with open(out_file, "w") as json_file:
            json.dump(self.__depth_histogram, json_file, indent=4)

        # Save depth histogram as png.
        plt.bar(list(self.__depth_histogram.keys()), list(self.__depth_histogram.values()))
        plt.xlim(min_depth, max_depth)
        plt.ylim(0, max(self.__depth_histogram.values()))
        plt.xlabel("Depth")
        plt.ylabel("Number of volumes")
        plt.title("Full depth histogram")
        out_file = os.path.join(self.__output_dir, self.__display_name + "-full_depth_histogram.png")
        plt.savefig(out_file, dpi=300)

        # Save a sub-range of depth histogram as png.
        plt.clf()
        plt.bar(list(self.__depth_histogram.keys()), list(self.__depth_histogram.values()))
        plt.xlim(10, 1000)
        plt.ylim(0, 5000)
        plt.xlabel("Depth")
        plt.ylabel("Number of volumes")
        plt.title("Sub depth histogram")
        out_file = os.path.join(self.__output_dir, self.__display_name + "-sub_depth_histogram.png")
        plt.savefig(out_file, dpi=300)

    def __create_images_index(self):
        self.__images_index = {}
        dcm_count = 0
        start_time = time.time()

        for root, dirs, files in os.walk(self.__images_dir):
            rel_root = os.path.relpath(root, self.__images_dir)

            for file in files:
                if dcm_count % 50000 == 0:
                    elapsed = time.time() - start_time
                    print("\033[2K", end="")
                    print(f"DICOM files indexed: {dcm_count}   Elapsed: {elapsed:.2f} sec", end="\r")

                if file.endswith(".dcm"):
                    if rel_root not in self.__images_index:
                        self.__images_index[rel_root] = [file]
                    else:
                        self.__images_index[rel_root].append(file)

                    dcm_count += 1

        print("")

    def __get_dicom_files_from_gcs(self, dicom_dir):
        client = storage.Client()
        bucket = client.bucket(self.__gcs_bucket_name)
        blobs = bucket.list_blobs(prefix=dicom_dir)
        dicom_files = [blob.name for blob in blobs if blob.name.endswith(".dcm")]
        return dicom_files

    def __get_dicom_files_from_disk(self, dicom_dir):
        # Find DICOM files in the images index if it exists.
        if self.__images_index is not None:
            rel_dicom_dir = os.path.relpath(dicom_dir, self.__images_dir)

            if not rel_dicom_dir in self.__images_index:
                logging.warning(f"DICOM directory '{dicom_dir}' not found in images index")
                return None

            return [os.path.join(dicom_dir, dicom_file) for dicom_file in self.__images_index[rel_dicom_dir]]

        # If images index does not exist, search DICOM files in the given folder.
        if not os.path.exists(dicom_dir):
            logging.warning(f"DICOM directory '{dicom_dir}' not found")
            return None

        dicom_files = []

        for filename in os.listdir(dicom_dir):
            if not filename.endswith('.dcm'):
                continue

            dicom_file = os.path.join(dicom_dir, filename)

            if not pydicom.misc.is_dicom(dicom_file):
                continue

            dicom_files.append(dicom_file)

        return dicom_files

    def __read_dicom_file_from_gcs(self, dicom_file):
        try:
            client = storage.Client()
            bucket = client.bucket(self.__gcs_bucket_name)
            blob = bucket.blob(dicom_file)
            content = BytesIO(blob.download_as_bytes())
            dataset = pydicom.dcmread(content)

            # Read DICOM tags.
            dicom_values = dicom_utils.read_dicom_tags_from_dataset(dataset, DICOM_TAGS_TO_READ)

            # Check image.
            if not dicom_utils.check_dicom_image_in_dataset(dataset):
                raise ValueError("No image in DICOM file")

            dicom_data = {"dicom_file": os.path.relpath(dicom_file, self.__images_dir), "dicom_values": dicom_values}
        except Exception as e:
            dicom_data = None

        return dicom_data

    def __read_dicom_files_from_gcs(self, dicom_files):
        with ThreadPoolExecutor(max_workers=self.__max_num_workers) as executor:
            futures = [executor.submit(self.__read_dicom_file_from_gcs, dicom_file) for dicom_file in dicom_files]
            return [future.result() for future in futures]

    def __read_dicom_files_from_disk(self, dicom_files):
        try:
            datasets = [pydicom.dcmread(dicom_file, defer_size=1024) for dicom_file in dicom_files]
            dicom_data = [{
                "dicom_file": os.path.relpath(dicom_files[index], self.__images_dir),
                "dicom_values": dicom_utils.read_dicom_tags_from_dataset(dataset, DICOM_TAGS_TO_READ)
            } for index, dataset in enumerate(datasets)]

            if not all(dicom_utils.check_dicom_image_in_dataset(dataset) for dataset in datasets):
                raise ValueError("No image in DICOM file")
        except:
            dicom_data = None

        return dicom_data

    def __validate_volume(self, volume, report_data):
        report_study_instance_uid = report_data["StudyInstanceUid"]
        report_series_instance_uid = report_data["series_instance_uid"]
        report_patient_id = report_data["PatientID"]
        report_patient_birth_date = report_data["PatientBirthDate"]
        report_study_date = report_data["StudyDate"]
        report_modality = report_data["Modality"].upper()

        for item in volume:
            try:
                # Filter out non-primary and localizer volumes.
                image_type = item["dicom_values"]["ImageType"]
                if "PRIMARY" not in image_type:
                    raise ValueError(f"Ignoring non-primary image type '{image_type}'")
                if "LOCALIZER" in image_type:
                    raise ValueError(f"Ignoring localizer image type '{image_type}'")

                dicom_sop_class_uid = item["dicom_values"]["SOPClassUID"]
                dicom_study_instance_uid = item["dicom_values"]["StudyInstanceUID"]
                dicom_series_instance_uid = item["dicom_values"]["SeriesInstanceUID"]
                dicom_patient_id = item["dicom_values"]["PatientID"]
                dicom_patient_birth_date = item["dicom_values"]["PatientBirthDate"]
                dicom_study_date = item["dicom_values"]["StudyDate"]
                dicom_modality = item["dicom_values"]["Modality"].upper()
                dicom_samples_per_pixel = item["dicom_values"]["SamplesPerPixel"]

                # Check if SOP Class UID is supported.
                if dicom_sop_class_uid not in DICOM_MODALITY_MAPPING:
                    raise ValueError(f"Unsupported modality '{pydicom.uid.UID(dicom_sop_class_uid).name}'")

                # Check if SOP Class UID is correct.
                if DICOM_MODALITY_MAPPING[dicom_sop_class_uid] != self.__modality:
                    raise ValueError(f"Incorrect modality '{pydicom.uid.UID(dicom_sop_class_uid).name}'")

                # Check consistency with report values.
                if dicom_study_instance_uid != report_study_instance_uid:
                    raise ValueError(f"Study Instance UID inconsistent (in report = {report_study_instance_uid}, in DICOM = {dicom_study_instance_uid})")
                if dicom_series_instance_uid != report_series_instance_uid:
                    raise ValueError(f"Series Instance UID inconsistent (in report = {report_series_instance_uid}, in DICOM = {dicom_series_instance_uid})")
                if dicom_patient_id != report_patient_id:
                    raise ValueError(f"Patient ID inconsistent (in report = {report_patient_id}, in DICOM = {dicom_patient_id})")
                if dicom_patient_birth_date != report_patient_birth_date:
                    raise ValueError(f"Patient birth date inconsistent (in report = {report_patient_birth_date}, in DICOM = {dicom_patient_birth_date})")
                if dicom_study_date != report_study_date:
                    raise ValueError(f"Study date inconsistent (in report = {report_study_date}, in DICOM = {dicom_study_date})")
                if dicom_modality != report_modality:
                    raise ValueError(f"Modality inconsistent (in report = {report_modality}, in DICOM = {dicom_modality})")

                # Ignore non-grayscale images.
                if dicom_samples_per_pixel != 1:
                    raise ValueError(f"Incorrect number of samples per pixel, should be 1 but got {dicom_samples_per_pixel} instead")

                # Ignore volumes with varying slice sizes.
                if item["dicom_values"]["Rows"] != volume[0]["dicom_values"]["Rows"] or item["dicom_values"]["Columns"] != volume[0]["dicom_values"]["Columns"]:
                    raise ValueError(f"Varying slice sizes detected (first slice: {volume[0]['dicom_values']['Columns']}x{volume[0]['dicom_values']['Rows']}, "
                                     f"current slice: {item['dicom_values']['Columns']}x{item['dicom_values']['Rows']})")

                # Check modality.
                if dicom_modality != self.__modality:
                    raise ValueError(f"Invalid modality '{dicom_modality}', it should be '{self.__modality}'")
            except Exception as e:
                err_msg = f"{e} in DICOM file"
                return False, err_msg

        return True, ""

    def __sort_volume(self, volume):
        sorted_volume = sorted(volume, key=lambda item: item["dicom_values"]["InstanceNumber"])
        return sorted_volume

    def __is_in_ascending_and_incrementing_order(self, volume):
        if len(volume) == 0:
            return True

        return all(volume[i]["dicom_values"]["InstanceNumber"] + 1 == volume[i + 1]["dicom_values"]["InstanceNumber"] for i in range(len(volume) - 1))

    def __load_generated_data(self):
        self.__pandas_full_dataset = pd.read_csv(self.__generated_data_file)
        # Make sure all the elements are converted from strings back to original Python types.
        self.__pandas_full_dataset["volume"] = self.__pandas_full_dataset["volume"].map(ast.literal_eval)
        self.__pandas_full_dataset["label"] = self.__pandas_full_dataset["label"].map(ast.literal_eval)

        # Determine if generated data contains DICOM or NIfTI volumes.
        if len(self.__pandas_full_dataset) > 0:
            volume = self.__pandas_full_dataset.iloc[0]["volume"]
            self.__use_nifti_files = "nifti_file" in volume

        self.__max_depth = 0

    def __split_data(self):
        if "split" in self.__pandas_full_dataset.columns:
            # Use existing split.
            print("Using existing split")
            self.__pandas_train_dataset = self.__pandas_full_dataset[self.__pandas_full_dataset["split"] == "train"]
            self.__pandas_validation_dataset = self.__pandas_full_dataset[self.__pandas_full_dataset["split"] == "validate"]
            self.__pandas_test_dataset = self.__pandas_full_dataset[self.__pandas_full_dataset["split"] == "test"]
            if len(self.__pandas_validation_dataset) == 0:
                self.__pandas_validation_dataset = self.__pandas_test_dataset
        else:
            # Generate split.
            print("Generating split")
            self.__pandas_train_dataset, temp = train_test_split(self.__pandas_full_dataset, test_size=0.2, random_state=self.__seed)
            self.__pandas_validation_dataset, self.__pandas_test_dataset = train_test_split(temp, test_size=0.5, random_state=self.__seed)

        print(f"Train dataset size: {len(self.__pandas_train_dataset)}")
        print(f"Validation dataset size: {len(self.__pandas_validation_dataset)}")
        print(f"Test dataset size: {len(self.__pandas_test_dataset)}")

    def __create_torch_datasets(self):
        self.__torch_train_dataset = GradientTorchDataset(pandas_dataframe=self.__pandas_train_dataset)
        self.__torch_validation_dataset = GradientTorchDataset(pandas_dataframe=self.__pandas_validation_dataset)
        self.__torch_test_dataset = GradientTorchDataset(pandas_dataframe=self.__pandas_test_dataset)


class GradientTorchDataset(Dataset):
    def __init__(self, pandas_dataframe):
        self.__pandas_dataframe = pandas_dataframe

    def __getitem__(self, idx):
        item = self.__pandas_dataframe.iloc[idx]
        return item

    def __len__(self):
        return len(self.__pandas_dataframe)
