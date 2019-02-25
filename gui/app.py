import sys
from PyQt5.QtWidgets import QDialog, QApplication
from hackathon_gui2 import Ui_Dialog
import os
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog,QLineEdit,QPushButton

class AppWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.det_img_brw_btn.clicked.connect(self.open_file)
        self.show()
        self.le = QLineEdit()
        self.le.setObjectName("det_img_txtbx")
        self.le.setText("Host")


    def open_file(self, main_window):
        # Select the file dialog design.
        dialog_style = QFileDialog.DontUseNativeDialog
        dialog_style |= QFileDialog.DontUseCustomDirectoryIcons

        # Open the file dialog to select an image file.
        self.file_chosen, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                          "JPEG (*.JPEG *.jpeg *.JPG *.jpg *.JPE *.jpe *JFIF *.jfif);; PNG (*.PNG *.png);; GIF (*.GIF *.gif);; Bitmap Files (*.BMP *.bmp *.DIB *.dib);; TIFF (*.TIF *.tif *.TIFF *.tiff);; ICO (*.ICO *.ico)",
                                                          options=dialog_style)

        # Show the path of the file chosen.
        if self.file_chosen:
            self.ui.det_img_txtbx.setText(self.file_chosen)
            # Change the text on the label to display the file path chosen.
        else:
            self.ui.det_img_txtbx.setText("No file was selected. Please select an image.")

    def button_click(self):
        # shost is a QString object
        shost = self.le.setText()
        print(shost)

        self.main_widget = QWidget(self)
        vboxMain = QVBoxLayout(self.main_widget)

        # Required Layouts
        grid0 = QGridLayout()
        grid1 = QGridLayout()
        grid2 = QGridLayout()
        grid3 = QGridLayout()
app = QApplication(sys.argv)
w = AppWindow()
w.show()
sys.exit(app.exec_())

