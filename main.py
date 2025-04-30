import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PySide6.QtCore import QFileInfo
from ui_mainwindow import Ui_MainWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.fileName.setVisible(False)
        self.ui.loadCSV.clicked.connect(self.open_dialog)

    def open_dialog(self):
        fname = QFileDialog.getOpenFileName(self, "Open File", "/", "CSV Files (*.csv)")
        self.ui.fileName.setVisible(True)
        fi = QFileInfo(fname[0])
        name = fi.fileName()
        print(fname)
        self.ui.fileName.setText(name)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

