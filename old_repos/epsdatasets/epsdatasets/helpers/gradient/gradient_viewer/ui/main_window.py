import os

from PyQt5 import uic
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIntValidator, QKeySequence
from PyQt5.QtWidgets import QApplication, QMainWindow, QShortcut, QMessageBox


class MainWindow(QMainWindow):
    load_button_clicked_signal = pyqtSignal()
    table_selection_changed_signal = pyqtSignal(int)
    table_selection_double_clicked_signal = pyqtSignal(int)
    search_edit_text_changed_signal = pyqtSignal(str)

    def __init__(self):
        super(MainWindow, self).__init__()

        ui_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main_window.ui")
        uic.loadUi(ui_file_path, self)
        self.resize(2500, 1500)

        self.__create_connections()
        self.__init_controls()

        self.__copy_shortcut = QShortcut(QKeySequence(Qt.CTRL + Qt.Key_C), self.table_widget)
        self.__copy_shortcut.activated.connect(self.__copy_to_clipboard)

    def get_max_results(self):
        try:
            max_results = int(self.max_results_edit.text())
        except Exception as e:
            raise ValueError("Invalid max results")
        return max_results

    def get_csv_file(self):
        csv_file = self.csv_file_edit.text()
        if not csv_file:
            raise ValueError("CSV file is empty")
        return csv_file

    def get_csv_column(self):
        csv_column = self.csv_column_edit.text()
        if not csv_column:
            raise ValueError("CSV column is empty")
        return csv_column

    def get_single_file(self):
        single_file = self.single_file_edit.text()
        if not single_file:
            raise ValueError("Single file is empty")
        return single_file

    def get_gcs_bucket(self):
        bucket = self.gcs_bucket_edit.text()
        if not bucket:
            raise ValueError("GCS bucket is empty")
        return bucket

    def get_gcs_dir(self):
        images_dir = self.gcs_dir_edit.text()
        if not images_dir:
            raise ValueError("GCS dir is empty")
        return images_dir

    def get_include_nifti_files(self):
        return self.include_nifti_check_box.isChecked()

    def get_include_dicom_files(self):
        return self.include_dicom_check_box.isChecked()

    def get_dicom_gcs_bucket(self):
        bucket = self.dicom_gcs_bucket_edit.text()
        if not bucket:
            raise ValueError("DICOM GCS bucket is empty")
        return bucket

    def get_dicom_gcs_images_dir(self):
        images_dir = self.dicom_gcs_images_dir_edit.text()
        if not images_dir:
            raise ValueError("DICOM GCS images dir is empty")
        return images_dir

    def show_status(self, message, timeout):
        self.statusbar.showMessage(message, timeout)

    def show_error(self, message, timeout):
        print(f"Error: {message}")
        QMessageBox.critical(self, "Error", f"{message}")

    def clear_status(self):
        self.statusbar.clearMessage()

    def clear_table(self):
        self.table_widget.setRowCount(0)

    def clear_preview(self):
        self.nifti_image_viewer.clear()
        self.dicom_image_viewer.clear()
        self.image_info_edit.clear()

    def show_nifti(self, numpy_image_array):
        self.nifti_image_viewer.load_image(numpy_image_array)

    def show_dicom(self, numpy_image_array):
        self.dicom_image_viewer.load_image(numpy_image_array)

    def show_image_info(self, image_info):
        self.image_info_edit.setText(image_info)

    def __create_connections(self):
        self.image_source_combo_box.currentIndexChanged.connect(self.image_source_combo_box_changed_handler)
        self.load_button.clicked.connect(self.load_button_clicked_handler)
        self.table_widget.selectionModel().selectionChanged.connect(self.table_selection_changed_handler)
        self.table_widget.cellDoubleClicked.connect(self.table_selection_double_clicked_handler)
        self.include_nifti_check_box.stateChanged.connect(self.include_nifti_check_box_state_changed_handler)
        self.include_dicom_check_box.stateChanged.connect(self.include_dicom_check_box_state_changed_handler)
        self.search_edit.textChanged.connect(self.search_edit_text_changed_handler)

    def __init_controls(self):
        self.image_source_combo_box.setCurrentIndex(0)
        self.image_source_combo_box_changed_handler(0)
        self.max_results_edit.setValidator(QIntValidator(bottom=1))
        self.include_nifti_check_box.setChecked(True)
        self.include_dicom_check_box.setChecked(False)

    def image_source_combo_box_changed_handler(self, index):
        if index == 0:
            # Load from bucket.
            self.max_results_label.setVisible(True)
            self.max_results_edit.setVisible(True)
            self.csv_file_label.setVisible(False)
            self.csv_file_edit.setVisible(False)
            self.csv_column_label.setVisible(False)
            self.csv_column_edit.setVisible(False)
            self.single_file_label.setVisible(False)
            self.single_file_edit.setVisible(False)
            self.gcs_bucket_label.setVisible(True)
            self.gcs_bucket_edit.setVisible(True)
            self.gcs_dir_label.setVisible(True)
            self.gcs_dir_edit.setVisible(True)
            self.include_nifti_check_box.setVisible(True)
            self.include_dicom_check_box.setVisible(True)
            self.dicom_gcs_bucket_label.setVisible(self.include_dicom_check_box.isChecked())
            self.dicom_gcs_bucket_edit.setVisible(self.include_dicom_check_box.isChecked())
            self.dicom_gcs_images_dir_label.setVisible(self.include_dicom_check_box.isChecked())
            self.dicom_gcs_images_dir_edit.setVisible(self.include_dicom_check_box.isChecked())
        elif index == 1:
            # Load from CSV.
            self.max_results_label.setVisible(False)
            self.max_results_edit.setVisible(False)
            self.csv_file_label.setVisible(True)
            self.csv_file_edit.setVisible(True)
            self.csv_column_label.setVisible(True)
            self.csv_column_edit.setVisible(True)
            self.single_file_label.setVisible(False)
            self.single_file_edit.setVisible(False)
            self.gcs_bucket_label.setVisible(True)
            self.gcs_bucket_edit.setVisible(True)
            self.gcs_dir_label.setVisible(True)
            self.gcs_dir_edit.setVisible(True)
            self.include_nifti_check_box.setVisible(True)
            self.include_dicom_check_box.setVisible(True)
            self.dicom_gcs_bucket_label.setVisible(self.include_dicom_check_box.isChecked())
            self.dicom_gcs_bucket_edit.setVisible(self.include_dicom_check_box.isChecked())
            self.dicom_gcs_images_dir_label.setVisible(self.include_dicom_check_box.isChecked())
            self.dicom_gcs_images_dir_edit.setVisible(self.include_dicom_check_box.isChecked())
        elif index == 2:
            # Load from Data Lake.
            self.max_results_label.setVisible(False)
            self.max_results_edit.setVisible(False)
            self.csv_file_label.setVisible(False)
            self.csv_file_edit.setVisible(False)
            self.csv_column_label.setVisible(False)
            self.csv_column_edit.setVisible(False)
            self.single_file_label.setVisible(False)
            self.single_file_edit.setVisible(False)
            self.gcs_bucket_label.setVisible(False)
            self.gcs_bucket_edit.setVisible(False)
            self.gcs_dir_label.setVisible(False)
            self.gcs_dir_edit.setVisible(False)
            self.include_nifti_check_box.setVisible(False)
            self.include_dicom_check_box.setVisible(False)
            self.dicom_gcs_bucket_label.setVisible(False)
            self.dicom_gcs_bucket_edit.setVisible(False)
            self.dicom_gcs_images_dir_label.setVisible(False)
            self.dicom_gcs_images_dir_edit.setVisible(False)
        elif index == 3:
            # Load from a single file.
            self.max_results_label.setVisible(False)
            self.max_results_edit.setVisible(False)
            self.csv_file_label.setVisible(False)
            self.csv_file_edit.setVisible(False)
            self.csv_column_label.setVisible(False)
            self.csv_column_edit.setVisible(False)
            self.single_file_label.setVisible(True)
            self.single_file_edit.setVisible(True)
            self.gcs_bucket_label.setVisible(True)
            self.gcs_bucket_edit.setVisible(True)
            self.gcs_dir_label.setVisible(True)
            self.gcs_dir_edit.setVisible(True)
            self.include_nifti_check_box.setVisible(True)
            self.include_dicom_check_box.setVisible(True)
            self.dicom_gcs_bucket_label.setVisible(self.include_dicom_check_box.isChecked())
            self.dicom_gcs_bucket_edit.setVisible(self.include_dicom_check_box.isChecked())
            self.dicom_gcs_images_dir_label.setVisible(self.include_dicom_check_box.isChecked())
            self.dicom_gcs_images_dir_edit.setVisible(self.include_dicom_check_box.isChecked())
        else:
            raise ValueError(f"Unsupported image source {index}")

    def load_button_clicked_handler(self):
        self.load_button_clicked_signal.emit()

    def table_selection_changed_handler(self, selected, deselected):
        selected_indexes = selected.indexes()
        if not selected_indexes:
            return

        index = selected_indexes[0].row()
        self.table_selection_changed_signal.emit(index)

    def table_selection_double_clicked_handler(self, row, column):
        self.table_selection_double_clicked_signal.emit(row)

    def include_nifti_check_box_state_changed_handler(self, state):
        pass

    def include_dicom_check_box_state_changed_handler(self, state):
        self.dicom_gcs_bucket_label.setVisible(self.include_dicom_check_box.isChecked())
        self.dicom_gcs_bucket_edit.setVisible(self.include_dicom_check_box.isChecked())
        self.dicom_gcs_images_dir_label.setVisible(self.include_dicom_check_box.isChecked())
        self.dicom_gcs_images_dir_edit.setVisible(self.include_dicom_check_box.isChecked())

    def search_edit_text_changed_handler(self):
        self.search_edit_text_changed_signal.emit(self.search_edit.text())

    def __copy_to_clipboard(self):
        selected_indexes = self.table_widget.selectedIndexes()
        if not selected_indexes:
            return

        row = selected_indexes[0].row()
        column = selected_indexes[0].column()
        item = self.table_widget.item(row, column)

        if item:
            clipboard = QApplication.clipboard()
            clipboard.setText(item.text())
            print(f"Copied to clipboard: {item.text()}")
