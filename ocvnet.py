import cv2 as cv


class NetIOs:
    def __init__(self):
        self.input = None
        self.output = None
        self.originalFrame = None
        self.processedFrame = None


class OCVNet:
    def __init__(self, model_path, config_path, framework, classes_path):
        self.net = cv.dnn.readNet(model_path, config_path, framework)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        self.out_names = self.net.getUnconnectedOutLayersNames()

        self.last_layer = self.net.getLayer(self.net.getLayerId(self.net.getLayerNames()[-1]))

        self.netIOs = NetIOs()
        self.confThreshold = 0.5
        self.nmsThreshold = 0.4

        with open(classes_path, 'r') as f:
            names = f.read().split('\n')
            self.classes = list(filter(None, names))

    def set_input(self, frame):
        self.netIOs.originalFrame = frame
        self.netIOs.input = cv.dnn.blobFromImage(
            frame,
            1.0,
            (416, 416),
            None,
            True,
            False,
            cv.CV_8U)
        self.net.setInput(self.netIOs.input, scalefactor=0.005)

    def forward(self):
        self.netIOs.output = self.net.forward(self.out_names)

    def get_output(self):
        return self.netIOs
