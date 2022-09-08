import sys, os
import cv2 as cv
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QTextEdit,QColorDialog, QWidget, QFrame, QLabel, QDialog, QApplication, QFileDialog, QPushButton, QVBoxLayout, QGridLayout, QGroupBox
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
import numpy as np
from pdf_reader import PDFReader
from datetime import datetime
import threading

class mythread(QtCore.QThread):
    capture_screen_signal = QtCore.pyqtSignal()
    change_text = QtCore.pyqtSignal(str)
    change_style = QtCore.pyqtSignal(str)
    change_scrollbar = QtCore.pyqtSignal(int)
    change_scrollbar_on = QtCore.pyqtSignal(int)

    def __init__(self, cap):
        QtCore.QThread.__init__(self, parent=None)
        self.scroll_on = False
        self.face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
        self.cp = CamProc()
        self.cap = cap
        ret, frame = self.cap.read()

    def run(self):

        cap = self.cap

        if not cap.isOpened():
            print("Camera didn't open")
            exit(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("video end")
                break

            gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.2, 18)
                dis = []
                for eye in eyes:
                    (ex, ey, ew, eh) = eye
                    roi = roi_color[ey:ey + eh, ex:ex + ew]
                    roig = roi_gray[ey:ey + eh, ex:ex + ew]
                    roig = cv.GaussianBlur(roig, (5, 5), 0)
                    roig = 255 - roig
                    center = (roi.shape[1] / 2, roi.shape[0] / 2)

                    circles = cv.HoughCircles(roig, cv.HOUGH_GRADIENT, 1, 100,
                                              param1=125, param2=13, minRadius=5, maxRadius=12)

                    if circles is not None:
                        rc = circles[0, 0]
                        circles = np.uint16(np.around(circles))
                        for c in circles[0, :]:
                            dis.append(rc[0] - center[0])
                            break


                if type(eyes) == np.ndarray:
                    self.cp.feed(True, self)
                else:
                    self.cp.feed(False, self)

                if len(dis) > 0:
                    mx = max(dis)
                    mn = min(dis)
                    # print(dis)
                    th = 7
                    th2 = 10
                    if mx > th and mx < th2:
                        if self.scroll_on:
                            print("left")
                            self.change_scrollbar.emit(1)
                    elif mn < -th and mn > -th2:
                        if self.scroll_on:
                            print('right')
                            self.change_scrollbar.emit(0)

                # for (ex, ey, ew, eh) in eyes:
                #     cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # cv.imshow('gray', gray)
            # cv.imshow('eyes', frame)

        cap.release()

    def toggle(self):
        if self.scroll_on:
            self.change_text.emit("OFF")
            self.change_style.emit("font: 14pt ""Times New Roman""; color: rgb(255, 0, 0);")
            self.scroll_on = False
        else:
            self.change_text.emit("ON")
            self.change_style.emit("font: 14pt ""Times New Roman""; color: rgb(0, 255, 0);")
            self.scroll_on = True
        self.change_scrollbar_on.emit(self.scroll_on)


class CamProc:
    sz = 100
    restT = 6
    openT = 3
    closeT = 4


    def __init__(self):
        self.h = []

    def ridx(self, skip, x):
        if skip >= len(self.h): return -1
        for i in reversed(range(skip+1)):
            if self.h[i] == x:
                return i
        return -1

    def feed(self, v, th):
        # print(v)
        self.h.append(v)
        if len(self.h) > self.sz:
            del self.h[0]

        if v:
            z1 = self.ridx(len(self.h)-1, 0)
            if z1 >= 0 and len(self.h) - z1 - 1 == self.restT:
                o1 = self.ridx(z1, 1)
                if o1 >= 0 and z1 - o1 <= self.closeT:
                    z2 = self.ridx(o1, 0)
                    if z2 >= 0 and o1 - z2 <= self.openT:
                        o2 = self.ridx(z2, 1)
                        if o2 >= 0 and z2 - o2 <= self.closeT:
                            z3 = self.ridx(o2, 0)
                            if o2 - z3 >= self.restT:
                                print("2 wings")
                                th.toggle()
                            elif o2 - z3 <= self.openT:
                                o3 = self.ridx(z3, 1)
                                if z3 - o3 <= self.closeT:
                                    z4 = self.ridx(o3, 0)
                                    if o3 - z4 >= self.restT:
                                        print('3 wings')
                                        th.capture_screen_signal.emit()


class MainWin(QDialog):
    bcB: QPushButton
    fcB: QPushButton
    ecB: QPushButton
    oB: QPushButton
    smB: QPushButton
    lmB: QPushButton
    bcF: QFrame
    fcF: QFrame
    sL: QLabel
    content: QWidget

    def __init__(self, parent=None):
        super(MainWin, self).__init__(parent)
        loadUi('main.ui', self)
        self.other = None
        self.scroll_on = False
        self.cur_path = ""
        self.smB.clicked.connect(self.sm_handler)
        self.lmB.clicked.connect(self.lm_handler)
        self.ecB.clicked.connect(self.ec_handler)
        self.bcB.clicked.connect(self.bc_handler)
        self.fcB.clicked.connect(self.fc_handler)
        self.oB.clicked.connect(self.open_handler)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        windowLayout = QVBoxLayout()
        windowLayout.addWidget(self.scroll_area)
        self.content.setLayout(windowLayout)
        self.scroll_speed = 150

    def change_scrollbar_handler(self, x):
        if not self.scroll_on:
            return
        nv = self.scroll_area.verticalScrollBar().value()
        if x:
            nv += self.scroll_speed
            if nv > self.scroll_area.verticalScrollBar().maximum():
                nv = self.scroll_area.verticalScrollBar().maximum()
        else:
            nv -= self.scroll_speed
            if nv < 0:
                nv = 0
        self.scroll_area.verticalScrollBar().setValue(nv)

    def change_scrollbar_on_handler(self, x):
        self.scroll_on = x

    def sm_handler(self):
        core.save_mark(self.scroll_area.verticalScrollBar().value())

    def lm_handler(self):
        x = core.get_mark()
        if x == -1: return
        self.scroll_area.verticalScrollBar().setValue(x)

    def capture_screen(self):
        print("scree capture")
        screen = QtWidgets.QApplication.primaryScreen()
        screenshot = screen.grabWindow(self.content.winId())
        screenshot.save('shot.jpg', 'jpg')



    def open_handler(self):
        self.file_name, _ = QFileDialog.getOpenFileName(self, "Open File",
                                                        "E:/ALL Esmael's Data/Projects/Python Projects/PDF Reader/pdfs",
                                                        "PDF (*.pdf)")
        self.bcF.setStyleSheet(" ")
        self.fcF.setStyleSheet(" ")
        l = self.file_name.split("/")
        name = l[-1]
        core.open_pdf(name)
        self.show()

    def show(self):
        v = self.scroll_area.verticalScrollBar().value()
        images = core.get_images()
        # print(images)

        # img_width = 22000
        img_width = 1280
        img_height = img_width * 1.4142

        w = QWidget()
        box = QVBoxLayout()
        box.setSpacing(2)
        box.setAlignment(Qt.AlignHCenter)
        for i in range(len(images)):
            img = QLabel()
            img.setFixedSize(img_width, img_height)
            img.setAlignment(Qt.AlignHCenter)
            bind_image(img, images[i])
            # bind_image(img, i)
            box.addWidget(img)
            # box.setStretch(i, 2)

        w.setLayout(box)
        self.scroll_area.setWidget(w)

        self.scroll_area.verticalScrollBar().setValue(v)

    def ec_handler(self):
        core.set_ec(self.ecB.isChecked())
        self.show()


    def bc_handler(self):
        selected_color = QColorDialog.getColor()
        if selected_color.isValid():
            self.bcF.setStyleSheet("QWidget { background-color: %s }" % selected_color.name())
            core.change_color([selected_color.blue(),selected_color.green(),selected_color.red()], None)
            self.show()

    def fc_handler(self):
        selected_color = QColorDialog.getColor()
        if selected_color.isValid():
            self.fcF.setStyleSheet("QWidget { background-color: %s }" % selected_color.name())
            core.change_color(None, [selected_color.blue(), selected_color.green(), selected_color.red()])
            self.show()


# global functions
def to_pixmap(a: np.ndarray):
    return QPixmap.fromImage(QImage(a.data, a.shape[1], a.shape[0], a.shape[1] * a.shape[2], QImage.Format_RGB888))


def bind_image(label: QtWidgets.QLabel, i):
    # pixel_map = QPixmap(core.cur_path + "/%s-page-%i.jpg" % (core.file_name[:-4], i))
    pixel_map = to_pixmap(i)
    pixel_map = pixel_map.scaled(label.width(), label.height(), Qt.KeepAspectRatio)
    label.setPixmap(pixel_map)


if __name__ == "__main__":

    core = PDFReader()
    app = QApplication(sys.argv)
    stacked_widget = QtWidgets.QStackedWidget()

    main_win = MainWin()

    thread = mythread(cv.VideoCapture(0))
    thread.capture_screen_signal.connect(main_win.capture_screen)
    # thread.toggle_roll.connect(main_win.toggle_scroll)
    thread.change_text.connect(main_win.sL.setText)
    thread.change_style.connect(main_win.sL.setStyleSheet)
    thread.change_scrollbar.connect(main_win.change_scrollbar_handler)
    thread.change_scrollbar_on .connect(main_win.change_scrollbar_on_handler)


    thread.start()

    main_win.other = thread

    stacked_widget.addWidget(main_win)
    stacked_widget.setGeometry(0, 0, 1000, 650)
    stacked_widget.setFixedWidth(1350)
    stacked_widget.setFixedHeight(650)
    stacked_widget.setWindowTitle("PDF Reader")
    stacked_widget.show()
    try:
        sys.exit(app.exec_())
    except:
        print("Exiting")
