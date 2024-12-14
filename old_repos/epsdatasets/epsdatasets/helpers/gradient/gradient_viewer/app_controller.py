import ast
import concurrent.futures
import os
import shutil
import time

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QTableWidgetItem, QApplication

from epsutils.dicom import dicom_utils
from epsutils.gcs import gcs_utils


class AppController(QObject):
    show_status_signal = pyqtSignal(str, int)
    show_error_signal = pyqtSignal(str, int)
    clear_status_signal = pyqtSignal()
    clear_table_signal = pyqtSignal()
    clear_preview_signal = pyqtSignal()
    show_nifti_signal = pyqtSignal(np.ndarray)
    show_dicom_signal = pyqtSignal(np.ndarray)
    show_image_info_signal = pyqtSignal(str)

    def __init__(self, main_window):
        QObject.__init__(self)

        self.__main_window = main_window
        self.__create_connections()

        self.__cache_dir = "./cache"
        os.makedirs(self.__cache_dir, exist_ok=True)

    def __del__(self):
        shutil.rmtree(self.__cache_dir)

    def __create_connections(self):
        self.__main_window.load_button_clicked_signal.connect(self.on_load_button_clicked)
        self.__main_window.table_selection_changed_signal.connect(self.on_table_selection_changed)
        self.__main_window.table_selection_double_clicked_signal.connect(self.on_table_selection_changed)

        self.show_status_signal.connect(self.__main_window.show_status)
        self.show_error_signal.connect(self.__main_window.show_error)
        self.clear_status_signal.connect(self.__main_window.clear_status)
        self.clear_table_signal.connect(self.__main_window.clear_table)
        self.clear_preview_signal.connect(self.__main_window.clear_preview)
        self.show_nifti_signal.connect(self.__main_window.show_nifti)
        self.show_image_info_signal.connect(self.__main_window.show_image_info)
        self.show_dicom_signal.connect(self.__main_window.show_dicom)

    def on_load_button_clicked(self):
        try:
            self.show_status_signal.emit("Loading images", 0)
            self.clear_table_signal.emit()
            QApplication.processEvents()

            index = self.__main_window.image_source_combo_box.currentIndex()

            if index == 0:
                # Load from bucket.
                self.__load_from_bucket()
            elif index == 1:
                # Load from CSV.
                self.__load_from_csv()
            elif index == 2:
                # Load from Data Lake.
                self.__load_from_data_lake()
            elif index == 3:
                # Load from a single file.
                self.__load_from_single_file()
            else:
                raise ValueError(f"Unsupported image source {index}")

            self.clear_status_signal.emit()

        except Exception as e:
            self.show_error_signal.emit(str(e), 10000)

    def on_table_selection_changed(self, index):
        try:
            self.show_status_signal.emit("Downloading data", 0)
            self.clear_preview_signal.emit()
            QApplication.processEvents()

            file_name = self.__main_window.table_widget.item(index, 0).text()
            self.__download_data(file_name)
            self.__show_data(file_name)

            self.clear_status_signal.emit()

        except Exception as e:
            self.show_error_signal.emit(f"{e}", 10000)

    def __load_from_bucket(self):
        gcs_bucket_name = self.__main_window.get_nifti_gcs_bucket()
        gcs_images_dir = self.__main_window.get_nifti_gcs_images_dir()
        max_results = self.__main_window.get_max_results()

        file_names = gcs_utils.list_files(gcs_bucket_name=gcs_bucket_name, gcs_dir=gcs_images_dir, max_results=max_results)
        file_names = [file_name for file_name in file_names if file_name.endswith(".nii.gz")]

        row_count = self.__main_window.table_widget.rowCount()
        for file_name in file_names:
            self.__main_window.table_widget.insertRow(row_count)
            item = QTableWidgetItem(file_name)
            self.__main_window.table_widget.setItem(row_count, 0, item)

    def __load_from_csv(self):
        columns = self.__main_window.get_csv_column().split(';')
        if len(columns) > 2:
            raise ValueError("CSV column can be either single name (e.g., 'volume') or composite of two names (e.g., 'volume;nifti_file')")

        column = columns[0]
        key = columns[1] if len(columns) == 2 else None

        df = pd.read_csv(self.__main_window.get_csv_file())
        items = df[column]
        row_count = self.__main_window.table_widget.rowCount()

        for item in items:
            file_name = item if key is None else ast.literal_eval(item)[key]
            self.__main_window.table_widget.insertRow(row_count)
            self.__main_window.table_widget.setItem(row_count, 0, QTableWidgetItem(file_name))

    def __load_from_data_lake(self):
        raise RuntimeError(f"Not implemented yet")

    def __load_from_single_file(self):
        file_name = self.__main_window.get_nifti_gcs_images_dir() + "/" + self.__main_window.get_single_file()

        row_count = self.__main_window.table_widget.rowCount()
        self.__main_window.table_widget.insertRow(row_count)
        self.__main_window.table_widget.setItem(row_count, 0, QTableWidgetItem(file_name))

    def __download_data(self, file_name):
        file_names = self.__generate_file_names(file_name)
        os.makedirs(file_names["data_dir"], exist_ok=True)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.__download_nifti, file_names["remote_nifti_file_name"], file_names["local_nifti_file_name"]),
                       executor.submit(self.__download_txt, file_names["remote_txt_file_name"], file_names["local_txt_file_name"]),
                       executor.submit(self.__download_dicom, file_names["remote_dicom_dir"], file_names["local_dicom_dir"])]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    def __download_nifti(self, remote_nifti_file_name, local_nifti_file_name):
        t1 = time.time()

        nifti_gcs_bucket_name = self.__main_window.get_nifti_gcs_bucket()
        gcs_utils.download_file(gcs_bucket_name=nifti_gcs_bucket_name,
                                gcs_file_name=remote_nifti_file_name,
                                local_file_name=local_nifti_file_name,
                                num_retries=None,  # Retry indefinitely.
                                show_warning_on_retry=True)

        if not os.path.exists(local_nifti_file_name):
            raise RuntimeError(f"Error downloading {remote_nifti_file_name}")

        t2 = time.time()
        print(f"NIfTI download time: {t2 - t1} sec")

    def __download_txt(self, remote_txt_file_name, local_txt_file_name):
        t1 = time.time()

        nifti_gcs_bucket_name = self.__main_window.get_nifti_gcs_bucket()
        gcs_utils.download_file(gcs_bucket_name=nifti_gcs_bucket_name,
                                gcs_file_name=remote_txt_file_name,
                                local_file_name=local_txt_file_name,
                                num_retries=None,  # Retry indefinitely.
                                show_warning_on_retry=True)

        if not os.path.exists(local_txt_file_name):
            raise RuntimeError(f"Error downloading {remote_txt_file_name}")

        t2 = time.time()
        print(f"TXT download time: {t2 - t1} sec")

    def __download_dicom(self, remote_dicom_dir, local_dicom_dir):
        if not self.__main_window.get_include_dicom_files():
            return

        t1 = time.time()

        dicom_gcs_bucket_name = self.__main_window.get_dicom_gcs_bucket()
        file_names = gcs_utils.list_files(gcs_bucket_name=dicom_gcs_bucket_name, gcs_dir=remote_dicom_dir)
        dicom_files = [file_name for file_name in file_names if file_name.endswith(".dcm")]

        with concurrent.futures.ProcessPoolExecutor() as dicoms_download_executor:
            futures = [dicoms_download_executor.submit(gcs_utils.download_file,
                                                       dicom_gcs_bucket_name,
                                                       dicom_file,
                                                       os.path.join(local_dicom_dir, os.path.basename(dicom_file)),
                                                       None,
                                                       True) for dicom_file in dicom_files]

            for future in concurrent.futures.as_completed(futures):
                future.result()

        t2 = time.time()
        print(f"DICOM download time: {t2 - t1} sec")

    def __show_data(self, file_name):
        file_names = self.__generate_file_names(file_name)

        # Show NIfTI file.
        sitk_image = sitk.ReadImage(file_names["local_nifti_file_name"])
        numpy_image_array = sitk.GetArrayFromImage(sitk_image)
        self.show_nifti_signal.emit(numpy_image_array)

        # Show image info.
        with open(file_names["local_txt_file_name"], "r") as file:
            image_info = file.read()
        self.show_image_info_signal.emit(image_info)

        # Show DICOM files.
        if self.__main_window.get_include_dicom_files():
            numpy_image_array = self.__load_dicom_files(file_names["local_dicom_dir"])
            self.show_dicom_signal.emit(numpy_image_array)

    def __generate_file_names(self, file_name):
        base_file_name = os.path.basename(file_name)
        data_dir = os.path.join(self.__cache_dir, base_file_name.replace(".nii.gz", ""))
        remote_nifti_file_name = file_name
        remote_txt_file_name = file_name.replace(".nii.gz", ".txt")
        local_nifti_file_name = os.path.join(data_dir, base_file_name)
        local_txt_file_name = os.path.join(data_dir, base_file_name.replace(".nii.gz", ".txt"))
        remote_dicom_dir = self.__main_window.get_dicom_gcs_images_dir() + "/" + file_name.replace("_", "/").replace(".nii.gz", "")
        local_dicom_dir = data_dir

        file_names = {
            "data_dir": data_dir,
            "remote_nifti_file_name": remote_nifti_file_name,
            "remote_txt_file_name": remote_txt_file_name,
            "local_nifti_file_name": local_nifti_file_name,
            "local_txt_file_name": local_txt_file_name,
            "remote_dicom_dir": remote_dicom_dir,
            "local_dicom_dir": local_dicom_dir
        }

        return file_names

    def __load_dicom_files(self, dicom_dir):
        dicom_datasets = []

        entries = os.listdir(dicom_dir)
        for entry in entries:
            if entry.endswith(".dcm"):
                dicom_file = os.path.join(dicom_dir, entry)
                dicom_dataset = pydicom.dcmread(dicom_file)
                dicom_datasets.append(dicom_dataset)

        # Sort DICOM files by the InstanceNumber.
        dicom_datasets = sorted(dicom_datasets, key=lambda dicom_dataset: dicom_dataset.InstanceNumber)

        # Get DICOM images.
        win = {"window_center": 0, "window_width": 0}
        dicom_images = [dicom_utils.get_dicom_image_from_dataset(dicom_dataset, win) for dicom_dataset in dicom_datasets]

        numpy_image_array = np.stack(dicom_images, axis=0)

        return numpy_image_array
