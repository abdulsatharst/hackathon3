#import sys
from PyQt5.QtCore import SIGNAL
#from PyQt5.QtGui import QDialog, QApplication, QPushButton, QLineEdit, QFormLayout
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from hackathon_gui2 import Ui_Dialog
import os
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog,QPushButton, QLineEdit, QFormLayout

class Form(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.le = QLineEdit()
        self.le.setObjectName("host")
        self.le.setText("Host")

        self.pb = QPushButton()
        self.pb.setObjectName("connect")
        self.pb.setText("Connect")

        layout = QFormLayout()
        layout.addWidget(self.le)
        layout.addWidget(self.pb)

        self.setLayout(layout)
        self.connect(self.pb, SIGNAL("clicked()"),self.button_click)
        self.setWindowTitle("Learning")
        self.show()

    def button_click(self):
        # shost is a QString object
        shost = self.le.text()
        print (shost)


app = QApplication(sys.argv)
w = AppWindow()
w.show()
sys.exit(app.exec_())
