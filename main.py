import sys
import yfinance as yf

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
        self.ui.predict.clicked.connect(self.show_data)


    def open_dialog(self):
        fname = QFileDialog.getOpenFileName(self, "Open File", "/", "CSV Files (*.csv)")
        self.ui.fileName.setVisible(True)
        fi = QFileInfo(fname[0])
        name = fi.fileName()
        self.ui.fileName.setText(name)


    def show_data(self):
        tick = "AAPL"
        p1 = financeAPI(tick)
        p1.showInfo()


class financeAPI:
    def __init__(self, ticker):
        self.ticker = yf.Ticker(ticker)

    def showInfo(self):
        his_data = self.ticker.history(period="1y")
        print(type(his_data))
        print(his_data)




def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()


