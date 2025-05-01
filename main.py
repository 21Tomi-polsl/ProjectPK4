import sys
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PySide6.QtCore import QFileInfo
from ui_mainwindow import Ui_MainWindow
from sklearn.linear_model import LinearRegression

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.fileName.setVisible(False)
        self.lista = []
        self.ui.loadCSV.clicked.connect(self.open_dialog)
        self.ui.predict.clicked.connect(self.collect_data)
        self.ui.trainModel.clicked.connect(self.start_train)


    def open_dialog(self):
        fname = QFileDialog.getOpenFileName(self, "Open File", "/", "CSV Files (*.csv)")
        self.ui.fileName.setVisible(True)
        fi = QFileInfo(fname[0])
        name = fi.fileName()
        self.ui.fileName.setText(name)


    def collect_data(self):
        tick = self.ui.tickerEdit.text()
        p1 = FinanceAPI(tick)
        self.lista = p1.gather_info()
        self.ui.label.setText("Your "+tick+" data is ready to be used!")

    def start_train(self):
        m1 = Model(self.lista)
        m1.linear_regression()


class FinanceAPI:
    def __init__(self, ticker):
        self.ticker = yf.Ticker(ticker)

    def gather_info(self):
        data = self.ticker.history(period="1d")
        close_list = data["Close"].tolist()
        open_list = data["Open"].tolist()
        high_list = data["High"].tolist()
        low_list = data["Low"].tolist()
        div = len(open_list)
        open = sum(open_list)/div
        close = sum(close_list)/div
        high = sum(high_list)/div
        low = sum(low_list)/div

        return [open, low, high, close]


class Model:
    def __init__(self, lista):
        self.lista = lista


    def linear_regression(self):
        x = [1,2,3,4]
        data = {
            'x':[1,2,3,4],
            'y':self.lista
        }
        df = pd.DataFrame(data)
        X = df[['x']]
        y = df[['y']]
        md = LinearRegression()
        md.fit(X, y)
        pred = md.predict(X)
        plt.scatter(X, y, color='blue')
        plt.plot(X, pred, color='red')
        plt.grid(True)
        plt.show()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()


