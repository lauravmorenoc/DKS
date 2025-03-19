import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QTimer

class SliderValuePrinter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Set up a simple GUI with a slider to adjust a value
        self.frequency = 1.0  # Initial value of the frequency
        
        # Create a slider for adjusting the frequency
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(10)  # Frequency times 10 (1.0 to 5.0)
        self.slider.setMaximum(50)  # Frequency times 10
        self.slider.setValue(int(self.frequency * 10))
        self.slider.valueChanged.connect(self.print_frequency)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.slider)

        # Container widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def print_frequency(self, value):
        self.frequency = value / 10.0
        print(f"Current Frequency: {self.frequency}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SliderValuePrinter()
    window.show()
    sys.exit(app.exec_())
