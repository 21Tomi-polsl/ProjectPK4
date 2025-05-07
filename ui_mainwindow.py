# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'untitled.ui'
##
## Created by: Qt User Interface Compiler version 6.9.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QLabel,
    QLineEdit, QMainWindow, QMenuBar, QPushButton,
    QSizePolicy, QStatusBar, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(600, 500)
        font = QFont()
        font.setFamilies([u"Bahnschrift"])
        MainWindow.setFont(font)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.load = QPushButton(self.centralwidget)
        self.load.setObjectName(u"load")
        self.load.setGeometry(QRect(360, 240, 111, 41))
        self.load.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(90, 20, 431, 41))
        font1 = QFont()
        font1.setFamilies([u"Bahnschrift"])
        font1.setPointSize(20)
        self.label.setFont(font1)
        self.optionsLabel = QLabel(self.centralwidget)
        self.optionsLabel.setObjectName(u"optionsLabel")
        self.optionsLabel.setGeometry(QRect(360, 130, 161, 41))
        font2 = QFont()
        font2.setFamilies([u"Bahnschrift"])
        font2.setPointSize(14)
        self.optionsLabel.setFont(font2)
        self.optionsLabel.setAutoFillBackground(False)
        self.optionsLabel.setScaledContents(True)
        self.optionsLabel.setWordWrap(True)
        self.trainModel = QPushButton(self.centralwidget)
        self.trainModel.setObjectName(u"trainModel")
        self.trainModel.setGeometry(QRect(240, 310, 131, 61))
        self.trainModel.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.errorBox = QCheckBox(self.centralwidget)
        self.errorBox.setObjectName(u"errorBox")
        self.errorBox.setGeometry(QRect(360, 170, 91, 20))
        self.graphBox = QCheckBox(self.centralwidget)
        self.graphBox.setObjectName(u"graphBox")
        self.graphBox.setGeometry(QRect(360, 190, 101, 20))
        self.exportBox = QCheckBox(self.centralwidget)
        self.exportBox.setObjectName(u"exportBox")
        self.exportBox.setGeometry(QRect(360, 210, 91, 20))
        self.modelLabel = QLabel(self.centralwidget)
        self.modelLabel.setObjectName(u"modelLabel")
        self.modelLabel.setGeometry(QRect(100, 210, 171, 41))
        self.modelLabel.setFont(font2)
        self.modelLabel.setAutoFillBackground(False)
        self.modelLabel.setScaledContents(True)
        self.modelLabel.setWordWrap(True)
        self.chooseModel = QComboBox(self.centralwidget)
        self.chooseModel.addItem("")
        self.chooseModel.addItem("")
        self.chooseModel.setObjectName(u"chooseModel")
        self.chooseModel.setGeometry(QRect(100, 250, 131, 31))
        self.chooseModel.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.chooseModel.setEditable(False)
        self.chooseModel.setMinimumContentsLength(3)
        self.fileLabel = QLabel(self.centralwidget)
        self.fileLabel.setObjectName(u"fileLabel")
        self.fileLabel.setGeometry(QRect(100, 130, 191, 41))
        self.fileLabel.setFont(font2)
        self.fileLabel.setAutoFillBackground(False)
        self.fileLabel.setScaledContents(True)
        self.fileLabel.setWordWrap(True)
        self.tickerEdit = QLineEdit(self.centralwidget)
        self.tickerEdit.setObjectName(u"tickerEdit")
        self.tickerEdit.setGeometry(QRect(100, 170, 113, 22))
        self.tickerEdit.setMaxLength(5)
        self.tickerEdit.setClearButtonEnabled(False)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 600, 18))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Stock Market Share Price Predictor", None))
        self.load.setText(QCoreApplication.translate("MainWindow", u"Load Data", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Stock Market Share Price Predictor", None))
        self.optionsLabel.setText(QCoreApplication.translate("MainWindow", u"Additional options", None))
        self.trainModel.setText(QCoreApplication.translate("MainWindow", u"Train Model", None))
        self.errorBox.setText(QCoreApplication.translate("MainWindow", u"Model Error", None))
        self.graphBox.setText(QCoreApplication.translate("MainWindow", u"Display graph", None))
        self.exportBox.setText(QCoreApplication.translate("MainWindow", u"Export to file", None))
        self.modelLabel.setText(QCoreApplication.translate("MainWindow", u"Choose model type", None))
        self.chooseModel.setItemText(0, QCoreApplication.translate("MainWindow", u"Regresja liniowa", None))
        self.chooseModel.setItemText(1, QCoreApplication.translate("MainWindow", u"LSTM", None))

        self.chooseModel.setCurrentText(QCoreApplication.translate("MainWindow", u"Regresja liniowa", None))
        self.chooseModel.setPlaceholderText("")
        self.fileLabel.setText(QCoreApplication.translate("MainWindow", u"Choose ticker symbol", None))
    # retranslateUi

