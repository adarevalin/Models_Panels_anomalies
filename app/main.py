import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap
from myapp_ui import Ui_MainWindow
from keras.models import load_model
import numpy as np
from PIL import Image

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.model = load_model(r'C:\\Users\\Public\\Proyectos\\modelo_panel_solar\\app\\modelo.h5')  # Ruta corregida

        # Cargar el logo
        self.ui.logoLabel.setPixmap(QPixmap(r'C:\\Users\\Public\\Proyectos\\modelo_panel_solar\\app\\logo.jpg').scaled(200, 50, aspectRatioMode=1))

        # Conectar el botón de carga de imagen
        self.ui.loadButton.clicked.connect(self.load_image)

        # Aplicar estilos
        self.apply_styles()

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F5F5;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 15px 32px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QLabel {
                font-size: 14px;
            }
        """)

    def load_image(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*);;Image Files (*.png;*.jpg;*.jpeg)", options=options)
        if fileName:
            pixmap = QPixmap(fileName)
            self.ui.imageLabel.setPixmap(pixmap.scaled(400, 300, aspectRatioMode=1))
            self.predict_image(fileName)

    def predict_image(self, image_path):
        image = Image.open(image_path).convert('L')
        image = image.resize((40, 24))
        input_data = np.array(image).reshape(1, 40, 24, 1)
        prediction = self.model.predict(input_data)
        self.ui.predictionLabel.setText(f'Prediction: {np.argmax(prediction)}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MyApp()
    main_window.show()
    sys.exit(app.exec_())
