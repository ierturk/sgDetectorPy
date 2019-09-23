from PySide2 import QtCore
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile, QObject, QTimer, QThread
from PySide2.QtWidgets import QPushButton, QLabel
from PySide2.QtGui import QImage, QPixmap
import cv2 as cv

from QueueFPS import QueueFPS
import queue

from ortnet import OrtNet


class CaptureThread(QThread):
    def __init__(self, url):
        QThread.__init__(self)
        QThread.currentThread().setPriority(QThread.HighPriority)
        self.capturedFrameQueue = QueueFPS()
        self.url = url
        self.capture = cv.VideoCapture(url)

    def run(self):
        while True:
            has_frame, frame = self.capture.read()
            if has_frame:
                self.capturedFrameQueue.put(frame)


class MainWindow(QObject):
    def __init__(self, ui_file, parent=None):
        super(MainWindow, self).__init__(parent)
        ui_file = QFile(ui_file)
        ui_file.open(QFile.ReadOnly)
        loader = QUiLoader()
        self.window = loader.load(ui_file)
        ui_file.close()

        self.lblPlayerRight = self.window.findChild(QLabel, 'lblPlayerRight')
        self.btnLoadRight = self.window.findChild(QPushButton, 'btnLoadRight')
        self.btnPlayRight = self.window.findChild(QPushButton, 'btnPlayRight')

        self.lblPlayerLeft = self.window.findChild(QLabel, 'lblPlayerLeft')
        self.btnLoadLeft = self.window.findChild(QPushButton, 'btnLoadLeft')
        self.btnPlayLeft = self.window.findChild(QPushButton, 'btnPlayLeft')

        self.captureTimerRight = QTimer(self)
        self.captureTimerLeft = QTimer(self)
        self.processTimerRight = QTimer(self)
        self.processTimerLeft = QTimer(self)
        self.viewerTimerRight = QTimer(self)
        self.viewerTimerLeft = QTimer(self)

        self.capturedFrameQueueRight = QueueFPS()
        self.capturedFrameQueueLeft = QueueFPS()
        self.resultQueueRight = QueueFPS()
        self.resultQueueLeft = QueueFPS()

        self.ortNet = \
            OrtNet("/home/ierturk/Work/REPOs/ssd/ssdIE/outputs/mobilenet_v2_ssd320_clk_trainval2019/model_040000.onnx")

        # self.captureRight = cv.VideoCapture("http://localhost:8080/2b50025c345db5308b9174da0a930295/mp4/pnOOUNJfOo/RightCam/s.mp4")
        # self.captureLeft = cv.VideoCapture("http://localhost:8080/2b50025c345db5308b9174da0a930295/mp4/pnOOUNJfOo/LeftCam/s.mp4")

        self.captureThreadRight = CaptureThread("http://localhost:8080/2b50025c345db5308b9174da0a930295/mp4/pnOOUNJfOo/RightCam/s.mp4")
        # self.captureThreadRight.start()
        self.captureThreadLeft = CaptureThread("http://localhost:8080/2b50025c345db5308b9174da0a930295/mp4/pnOOUNJfOo/LeftCam/s.mp4")
        # self.captureThreadLeft.start()

        # self.captureTimerRight.timeout.connect(lambda: self.capture_frame(True))
        # self.captureTimerRight.start()
        # self.captureTimerLeft.timeout.connect(lambda: self.capture_frame(False))
        # self.captureTimerLeft.start()

        self.processTimerRight.timeout.connect(lambda: self.process_frame(True))
        self.processTimerRight.start()
        self.processTimerLeft.timeout.connect(lambda: self.process_frame(False))
        self.processTimerLeft.start()

        self.viewerTimerRight.timeout.connect(lambda: self.update_gui(True))
        self.viewerTimerRight.start()
        self.viewerTimerLeft.timeout.connect(lambda: self.update_gui(False))
        self.viewerTimerLeft.start()

        self.window.show()

    def capture_frame(self, side):
        if side is True:
            has_frame, frame = self.captureRight.read()
            if has_frame:
                self.capturedFrameQueueRight.put(frame)
        else:
            has_frame, frame = self.captureLeft.read()
            if has_frame:
                self.capturedFrameQueueLeft.put(frame)

    def process_frame(self, side):
        frame = None
        if side:
            try:
                # frame = self.capturedFrameQueueRight.get_nowait()
                frame = self.captureThreadRight.capturedFrameQueue.get_nowait()
                self.capturedFrameQueueRight.queue.clear()
            except queue.Empty:
                pass

            if not frame is None:
                self.ortNet.set_input(frame, True)
                self.ortNet.forward(True)
                self.resultQueueRight.put(self.ortNet.get_processed_frame(True))
        else:
            try:
                # frame = self.capturedFrameQueueLeft.get_nowait()
                frame = self.captureThreadLeft.capturedFrameQueue.get_nowait()
                self.capturedFrameQueueLeft.queue.clear()
            except queue.Empty:
                pass

            if not frame is None:
                self.ortNet.set_input(frame, False)
                self.ortNet.forward(False)
                self.resultQueueLeft.put(self.ortNet.get_processed_frame(False))

    def update_gui(self, side):
        frame = None
        if side is True:
            try:
                frame = self.resultQueueRight.get_nowait()
            except queue.Empty:
                pass

            if not frame is None:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                pix = QPixmap.fromImage(img)
                self.lblPlayerRight.setPixmap(pix)
                self.lblPlayerRight.setScaledContents(True)
        else:
            try:
                frame = self.resultQueueLeft.get_nowait()
            except queue.Empty:
                pass

            if not frame is None:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                pix = QPixmap.fromImage(img)
                self.lblPlayerLeft.setPixmap(pix)
                self.lblPlayerLeft.setScaledContents(True)
