import onnxruntime as rt
import cv2 as cv
from dataset import letterbox


class NetIOs:
    def __init__(self):
        self.input = None
        self.output = None
        self.originalFrame = None
        self.processedFrame = None


class OrtNet:
    def __init__(self, model_path, classes_path):

        self.sessionOptions = rt.SessionOptions()
        self.sessionOptions.set_graph_optimization_level(2)
        # self.sessionOptions.session_log_severity_level = 4
        self.sess = rt.InferenceSession(model_path, self.sessionOptions)
        self.input_name = self.sess.get_inputs()[0].name
        self.netIOs = NetIOs()

        self.confThreshold = 0.3
        self.nmsThreshold = 0.5

        with open(classes_path, 'r') as f:
            names = f.read().split('\n')
            self.classes = list(filter(None, names))

    def set_input(self, frame):
        self.netIOs.originalFrame = frame
        frame = letterbox(frame, (320, 320), mode='rect')[0]
        self.netIOs.input = cv.dnn.blobFromImage(
            frame,
            1/255,
            None,
            None,
            True,
            False,
            cv.CV_32F)

    def forward(self):
        self.netIOs.output = self.sess.run(None, {self.input_name: self.netIOs.input})

    def get_output(self):
        return self.netIOs
