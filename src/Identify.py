from PyQt6 import QtCore, QtGui, QtWidgets
import cv2
from ultralytics import YOLO
import numpy as np
from PyQt6.QtGui import QPixmap, QImage


model = YOLO('../data/best3.pt')

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 700)  
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        
        self.main_layout = QtWidgets.QVBoxLayout(self.centralwidget)

        
        self.label = QtWidgets.QLabel(parent=self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(30)  
        font.setBold(True)
        font.setWeight(100)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setObjectName("label")
        self.label.setStyleSheet("color: black;")  
        self.main_layout.addWidget(self.label)


        # QLabel to display images or videos
        self.imageLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.imageLabel.setObjectName("imageLabel")
        self.imageLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.imageLabel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.main_layout.addWidget(self.imageLabel)

        # Create a horizontal layout for buttons
        self.button_layout = QtWidgets.QHBoxLayout()

        # Buttons
        self.pushButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.detect_image)
        self.button_layout.addWidget(self.pushButton)

        self.pushButton_2 = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.detect_video)
        self.button_layout.addWidget(self.pushButton_2)

        self.pushButton_3 = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.detect_webcam)
        self.button_layout.addWidget(self.pushButton_3)

        # Add button layout to the main layout
        self.main_layout.addLayout(self.button_layout)

        # Subtitle label
        self.label_2 = QtWidgets.QLabel(parent=self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.main_layout.addWidget(self.label_2)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 26))
        self.menubar.setObjectName("menubar")
        self.menuok = QtWidgets.QMenu(parent=self.menubar)
        self.menuok.setObjectName("menuok")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuok.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Apply stylesheets
        self.apply_stylesheets()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Vehicle Brand Identify"))
        self.label.setText(_translate("MainWindow", ""))
        self.label_2.setText(_translate("MainWindow", "Functions"))
        self.pushButton.setText(_translate("MainWindow", "Image"))
        self.pushButton_2.setText(_translate("MainWindow", "Video"))
        self.pushButton_3.setText(_translate("MainWindow", "Webcam"))
        self.menuok.setTitle(_translate("MainWindow", "Identify"))

    def apply_stylesheets(self):
        self.pushButton.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 10px 20px;
                color: white;
                background-color: #4CAF50;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        self.pushButton_2.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 10px 20px;
                color: white;
                background-color: #2196F3;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)

        self.pushButton_3.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 10px 20px;
                color: white;
                background-color: #f44336;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)

        self.centralwidget.setStyleSheet("""
            QLabel {
                color: white;
                background-color: white;
            }
            QWidget {
                font-size: 12px;
            }
            QMainWindow {
                background-color: #333;
            }
        """)

    def detect_image(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Chọn ảnh", "", "Image Files (*.jpg *.jpeg *.png)")
        if file_path:
            img = cv2.imread(file_path)
            results = model(img)
            self.display_results(img)

    def detect_video(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Chọn video", "", "Video Files (*.mp4 *.avi)")
        if file_path:
            cap = cv2.VideoCapture(file_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame)
                self.display_results(frame, video_mode=True)
            cap.release()

    def detect_webcam(self):
        cap = cv2.VideoCapture(0)  # Use default webcam
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            self.display_results(frame, video_mode=True)
        cap.release()

    def display_results(self, img, video_mode=False):
        for result in model(img):
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                # Vẽ bounding box và nhãn lên ảnh
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, f'{class_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Resize QLabel to fit the central widget's size
        self.imageLabel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.imageLabel.setPixmap(pixmap.scaled(self.imageLabel.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation))
        self.imageLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        if video_mode:
            self.imageLabel.repaint()  # Update interface for video frames

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
