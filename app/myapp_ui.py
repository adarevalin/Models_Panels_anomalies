# myapp_ui.py

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(400, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Logo
        self.logoLabel = QtWidgets.QLabel(self.centralwidget)
        self.logoLabel.setGeometry(QtCore.QRect(50, 10, 300, 50))  # Ajusta el tamaño del logo
        self.logoLabel.setAlignment(QtCore.Qt.AlignCenter)  # Centra el contenido
        self.logoLabel.setObjectName("logoLabel")

        self.loadButton = QtWidgets.QPushButton(self.centralwidget)
        self.loadButton.setGeometry(QtCore.QRect(100, 80, 200, 50))  # Centra el botón
        self.loadButton.setObjectName("loadButton")

        self.imageLabel = QtWidgets.QLabel(self.centralwidget)
        self.imageLabel.setGeometry(QtCore.QRect(50, 150, 300, 300))  # Centra la imagen
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)  # Centra el contenido
        self.imageLabel.setObjectName("imageLabel")

        self.predictionLabel = QtWidgets.QLabel(self.centralwidget)
        self.predictionLabel.setGeometry(QtCore.QRect(50, 470, 300, 50))  # Centra la etiqueta de predicción
        self.predictionLabel.setAlignment(QtCore.Qt.AlignCenter)  # Centra el contenido
        self.predictionLabel.setObjectName("predictionLabel")

        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.loadButton.setText(_translate("MainWindow", "Load Image"))
        self.imageLabel.setText(_translate("MainWindow", ""))
        self.predictionLabel.setText(_translate("MainWindow", ""))
