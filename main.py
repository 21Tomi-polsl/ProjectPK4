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
        self.tick = ""
        #Button functions
        self.ui.loadCSV.clicked.connect(self.open_dialog)
        self.ui.load.clicked.connect(self.collect_data)
        self.ui.trainModel.clicked.connect(self.start_train)
        #Checkbox functions

#File dialog opener function
    def open_dialog(self):
        fname = QFileDialog.getOpenFileName(self, "Open File", "/", "CSV Files (*.csv)")
        self.ui.fileName.setVisible(True)
        fi = QFileInfo(fname[0])
        name = fi.fileName()
        self.ui.fileName.setText(name)


    def collect_data(self):
        self.tick = self.ui.tickerEdit.text()
        try:
            p1 = FinanceAPI(self.tick)
            self.lista = p1.gather_info()
            self.ui.label.setText("Your "+self.tick+" data is ready to be used!")
        except:
            self.ui.label.setText("Input data is incorrect, try again!")



    def start_train(self):
        m1 = Model(self.lista)
        m1.prepare_linreg()
        predicted_value = m1.linear_regression()
        self.ui.label.setText(self.tick+" predicted value is "+ str(predicted_value))

#Yfinance API handling class
class FinanceAPI:
    def __init__(self, ticker):
        self.ticker = yf.Ticker(ticker)

    def gather_info(self):
        data = self.ticker.history(period="7d")

        close_list = data["Close"].tolist()
        open_list = data["Open"].tolist()
        high_list = data["High"].tolist()
        low_list = data["Low"].tolist()

        return [open_list, low_list, high_list, close_list]

#ML Model class
class Model:
    def __init__(self, lista):
        #List for linear regression
        self.lista = lista
        self.open = lista[0]
        self.close = lista[3]
        self.high = lista[2]
        self.low = lista[1]

        self.openHigh=0
        self.openLow=0
        self.closeHigh=0
        self.closeLow=0

        self.linRegList = []


    def prepare_linreg(self):

        div = len(self.open)
        self.open = sum(self.open) / div
        self.close = sum(self.close) / div
        self.high = sum(self.high) / div
        self.low = sum(self.low) / div

        self.openHigh = (self.open+self.high)/2
        self.closeHigh = (self.close+self.high)/2

        self.openLow = (self.open+self.low)/2
        self.closeLow = (self.close+self.low)/2

        self.linRegList = [self.open, self.openLow, self.openHigh, self.low, self.high, self.closeLow, self.closeHigh, self.close]


    def linear_regression(self):
        data = {
            'x':[1,2,3,4,5,6,7,8],
            'y':self.linRegList
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
        predicted_value = pred[-1]
        wynik = float(predicted_value[0])
        return round(wynik, 2)

#Main function for running the app
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()


