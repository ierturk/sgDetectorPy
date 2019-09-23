import onnxruntime as rt
import cv2 as cv
import numpy as np
import json


class NetIOs:
    def __init__(self):
        self.input = None
        self.output = None
        self.originalFrame = None
        self.processedFrame = None


class OrtNet:
    def __init__(self, model_path):
        self.sessionOptions = rt.SessionOptions()
        self.sessionOptions.set_graph_optimization_level(2)
        self.sess = rt.InferenceSession(model_path, self.sessionOptions)
        self.input_name = self.sess.get_inputs()[0].name
        self.netIOsRight = NetIOs()
        self.netIOsLeft = NetIOs()

        self.confThreshold = 0.5
        self.nmsThreshold = 0.4

        self.classes = None
        with open("/home/ierturk/Work/REPOs/ssd/yoloData/clk/train.json", 'rt') as f:
            json_data = json.load(f)
            self.classes = [cat['name'] for cat in json_data['categories']]

    def set_input(self, in_frame, side):
        if side:
            self.netIOsRight.originalFrame = in_frame
            self.netIOsRight.input = cv.dnn.blobFromImage(
                in_frame,
                1.0,
                (320, 320),
                (123, 117, 104),
                True,
                False,
                cv.CV_32F)
        else:
            self.netIOsLeft.originalFrame = in_frame
            self.netIOsLeft.input = cv.dnn.blobFromImage(
                in_frame,
                1.0,
                (320, 320),
                (123, 117, 104),
                True,
                False,
                cv.CV_32F)

    def forward(self, side):
        if side:
            self.netIOsRight.output = \
                self.sess.run(None, {self.input_name: self.netIOsRight.input})
        else:
            self.netIOsLeft.output = \
                self.sess.run(None, {self.input_name: self.netIOsLeft.input})

    def get_processed_frame(self, side):
        if side:
            return self.postprocess(self.netIOsRight.originalFrame, self.netIOsRight.output)
        else:
            return self.postprocess(self.netIOsLeft.originalFrame, self.netIOsLeft.output)

    def postprocess(self, frame, outs):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        batches_scores, batches_boxes = outs
        det = np.where(batches_scores[0, :, 1:] > self.confThreshold)
        class_ids = det[1].tolist()
        confidences = batches_scores[0, det[0], det[1] + 1].tolist()
        boxes = [[
            int(box[0] * frame_width),
            int(box[1] * frame_height),
            int((box[2] - box[0]) * frame_width) + 1,
            int((box[3] - box[1]) * frame_height) + 1,
        ] for box in batches_boxes[0, det[0], :].tolist()]

        indices = cv.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.draw_pred(frame, class_ids[i], confidences[i], left, top, left + width, top + height)

        return frame

    def draw_pred(self, frame, class_id, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))
        label = '%.2f' % conf

        # Print a label of class.
        if self.classes:
            assert (class_id < len(self.classes))
            label = '%s: %s' % (self.classes[class_id], label)

        label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, label_size[1])
        cv.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line), (255, 255, 255),
                     cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        return frame
