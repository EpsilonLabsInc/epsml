import os

import numpy as np
from PyQt5 import uic
from PyQt5.QtCore import Qt, QObject, QEvent, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QWidget


class SliderEventFilter(QObject):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress or event.type() == QEvent.KeyRelease:
            if event.key() == Qt.Key_Control:
                self.widget.ctrl_pressed = (event.type() == QEvent.KeyPress)

        return super().eventFilter(obj, event)


class ImageViewer(QWidget):
    sync_mode_sliding_signal = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        ui_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image_viewer.ui")
        uic.loadUi(ui_file_path, self)

        self.__event_filter = SliderEventFilter(self)
        self.installEventFilter(self.__event_filter)

        self.ctrl_pressed = False
        self.__numpy_image_array = None

        self.__create_connections()
        self.clear()

    def __create_connections(self):
        self.slider.valueChanged.connect(self.on_slider_position_changed)

    def clear(self):
        self.__numpy_image_array = None

        self.preview_label.clear()
        self.current_slice_label.clear()
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)

    def load_image(self, numpy_image_array):
        self.__numpy_image_array = numpy_image_array
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.get_num_slices() - 1)
        self.slider.setValue(0)
        self.on_slider_position_changed(0)

    def change_position(self, position):
        self.slider.setValue(position)

    def get_num_slices(self):
        return self.__numpy_image_array.shape[0] if self.__numpy_image_array is not None else 0

    def on_slider_position_changed(self, position):
        if self.__numpy_image_array is None:
            return

        self.__show_slice(position)
        self.current_slice_label.setText(f"{position + 1}/{self.get_num_slices()}")

        if self.ctrl_pressed:
            self.sync_mode_sliding_signal.emit(position)

    def __show_slice(self, slice_index):
        if self.__numpy_image_array is None:
            return

        image = self.__numpy_image_array[slice_index, :, :]
        image = (image - image.min()) / (image.max() - image.min()) * 255
        image = image.astype(np.uint8)
        height, width = image.shape
        num_bytes_per_line = width
        q_image = QImage(image.data, width, height, num_bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        self.preview_label.setPixmap(pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
