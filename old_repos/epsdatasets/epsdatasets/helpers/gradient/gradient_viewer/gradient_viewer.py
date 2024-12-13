import sys

from PyQt5.QtWidgets import QApplication

from app_controller import AppController
from ui.main_window import MainWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    app_controller = AppController(main_window=main_window)
    main_window.show()
    app.exec()
