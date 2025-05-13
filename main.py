#P.1 used modules
import sys
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import joblib
import re

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel

from ui_mainwindow import Ui_MainWindow

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

#P.3 Used filesystem for changing env variables (required for quality numerical predictions)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

from keras.models import Sequential
from keras.layers import LSTM, Dense


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.lista = []
        self.tick = ""
        #Button functions
        self.ui.load.clicked.connect(self.collect_data)
        self.ui.trainModel.clicked.connect(self.start_train)
        self.ui.helpButton.clicked.connect(self.show_help)
        #Checkbox functions
        self.display_graph = False
        self.display_error = False
        self.save_model = False

#Function for displaying error on screen
    def print_error(self, error):
        self.ui.calculatedError.setText("Model error: "+str(error))

#Function for collecting and parsing input data from the user
    def collect_data(self):
        self.tick = self.ui.tickerEdit.text()
        self.display_graph = self.ui.graphBox.isChecked()
        self.display_error = self.ui.errorBox.isChecked()
        self.save_model = self.ui.exportBox.isChecked()
        try:
            p1 = FinanceAPI(self.tick)
            self.lista = p1.gather_info()
            self.ui.label.setText("Your "+self.tick+" data is ready to be used!")
        except:
            self.ui.label.setText("Input data is incorrect, try again!")

#Function for model training after input
    def start_train(self):
        mod = self.ui.chooseModel.currentText()
        #P.2 Used regex
        if bool(re.search("Regresja liniowa", mod)):
            m1 = Model(self.lista)
            m1.prepare_linreg()
            predicted_values = m1.linear_regression(self.display_graph, self.display_error, self.save_model)
            self.ui.label.setText(self.tick+" predicted value is "+ str(predicted_values[0]))

            if predicted_values[1] != 0:
                self.print_error(predicted_values[1])

        else:
            self.ui.label.setText("The model is being trained, please wait")
            m1 = Model(self.lista)
            list = m1.prepare_lstm()
            predicted_values = m1.train_LSTM(list, self.display_graph, self.display_error, self.save_model)
            self.ui.label.setText(self.tick + " predicted value is " + str(predicted_values[0]))

            if predicted_values[1] != 0:
                self.print_error(predicted_values[1])

#Function for displaying instruction manual
    def show_help(self):
        self.w = HelpWindow()
        self.w.show()

#Class for 2nd window/help window initialize and display
class HelpWindow(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        instruction = ("1. Type in stock ticker\n"
                       "2. Choose model type\n"
                       "3. Add helper attributes(optional)\n"
                       "4. Press load data to download stock info\n"
                       "5. Wait for the announcement\n"
                       "6. Press train model and wait for your prediction!")
        self.label = QLabel(instruction)
        layout.addWidget(self.label)
        self.setLayout(layout)


#Yfinance API handling class
class FinanceAPI:
    def __init__(self, ticker):
        self.ticker = yf.Ticker(ticker)

#Function for gathering information from yahoo
    def gather_info(self):
        data = self.ticker.history(period="1y")

        close_list = data["Close"].tolist()
        open_list = data["Open"].tolist()
        high_list = data["High"].tolist()
        low_list = data["Low"].tolist()
        volume_list = data["Volume"].tolist()

        return [open_list, low_list, high_list, close_list, volume_list]

#ML Model class
class Model:
    def __init__(self, lista):
        #List for linear regression
        self.lista = lista
        self.open = lista[0]
        self.low = lista[1]
        self.high = lista[2]
        self.close = lista[3]
        self.volume = lista[4]

        self.openHigh=0
        self.openLow=0
        self.closeHigh=0
        self.closeLow=0

        self.linRegList = []
        self.lstmlist = {}


#Method for preparing variables for linear regression
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


    def linear_regression(self, displayGraph, displayError, saveToFile):
        data = {
            'x':[1,2,3,4,5,6,7,8],
            'y':self.linRegList}

        df = pd.DataFrame(data)
        X = df[['x']]
        y = df[['y']]

        md = LinearRegression()
        md.fit(X, y)
        pred = md.predict(X)

        if displayGraph:
            plt.scatter(X, y, color='blue')
            plt.plot(X, pred, color='red')
            plt.grid(True)
            plt.show()

        if displayError:
            mae = mean_absolute_error(y, pred)
            print(mae)
        else:
            mae = 0

        if saveToFile:
            joblib.dump(md, 'model.pkl')

        predicted_value = pred[-1]
        result = float(predicted_value[0])
        return [round(result, 2), round(mae, 2)]

    def prepare_lstm(self):
        self.lstmlist = {
            'Open':self.open,
            'High':self.high,
            'Low':self.low,
            'Close':self.close,
            'Volume':self.volume
        }
        df = pd.DataFrame(self.lstmlist)

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)


        def create_sequences(data, seq_len):
            X = []
            y = []

            for i in range(len(data)-seq_len):
                X.append(data[i:i+seq_len])
                y.append(data[i+seq_len, 3])
            return np.array(X), np.array(y)

        seq_length = 5
        X, y = create_sequences(scaled_data, seq_length)
        return [X, y, scaled_data, scaler, df]

    def train_LSTM(self, list, displayGraph, displayError, saveToFile):
        model = Sequential()
        X = list[0]
        y = list[1]
        scaled_data = list[2]
        scaler = list[3]
        df = list[4]

        model.add(LSTM(30, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=150, batch_size=32)

        y_pred = model.predict(X)

        y_pred_full = np.zeros((len(y_pred), scaled_data.shape[1]))
        y_pred_full[:, 3] = y_pred[:, 0]
        y_pred_original = scaler.inverse_transform(y_pred_full)[:, 3]


        y_true_full = np.zeros((len(y), scaled_data.shape[1]))
        y_true_full[:, 3] = y
        y_true_original = scaler.inverse_transform(y_true_full)[:, 3]


        if displayGraph:
            plt.plot(y_true_original, label="True")
            plt.plot(y_pred_original, label="Predict")
            plt.xlabel("Sample")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)
            plt.show()

        last_days = df[-50:]
        scale_data = scaler.transform(last_days)
        inpt = np.array([scale_data])

        new_pred = model.predict(inpt)
        new_pred_full = np.zeros((1, scale_data.shape[1]))
        new_pred_full[0,3] = new_pred[0,0]


        if displayError:
            mse = mean_squared_error(y_true_original, y_pred_original)
            print(mse)
        else:
            mse = 0

        if saveToFile:
            joblib.dump(model, 'model.pkl')

        predicted_price = scaler.inverse_transform(new_pred_full)[0,3]

        return [round(predicted_price, 2), round(mse, 2)]

#Main function for running the app
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()


