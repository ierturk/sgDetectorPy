import onnxruntime as rt
import cv2 as cv
import json


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

        self.confThreshold = 0.5
        self.nmsThreshold = 0.4

        self.classes = None
        with open(classes_path, 'rt') as f:
            json_data = json.load(f)
            self.classes = [cat['name'] for cat in json_data['categories']]

    def set_input(self, frame):
        self.netIOs.originalFrame = frame
        self.netIOs.input = cv.dnn.blobFromImage(
            frame,
            1.0,
            (320, 320),
            (123, 117, 104),
            True,
            False,
            cv.CV_32F)

    def forward(self):
        self.netIOs.output = self.sess.run(None, {self.input_name: self.netIOs.input})

    def get_output(self):
        return self.netIOs
