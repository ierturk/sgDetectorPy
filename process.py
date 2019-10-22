from threading import Thread
import cv2 as cv
import numpy as np
from QueueFPS import QueueFPS
import queue
import time
from ocvnet import OCVNet


class CaptureThread(Thread):
    def __init__(self, cap):
        super().__init__()
        self.queue = QueueFPS()
        self.cap = cv.VideoCapture(cap)
        self.process = False
        self.frame_count = self.cap.get(cv.CAP_PROP_FRAME_COUNT)
        self.current_frame = 0

    def run(self):
        while self.process:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, self.current_frame)
            try:
                has_frame, frame = self.cap.read()
                if not has_frame:
                    break
                self.queue.put(frame)
                time.sleep(0.04)
            except cv.error as e:
                print("cv2.error:", e)
            except Exception as e:
                print("Exception:", e)


class ProcessChannel(Thread):
    def __init__(self, cap):
        super().__init__()
        self.queue = QueueFPS()
        self.process = False
        self.capture = CaptureThread(cap)
        self.net = \
            OCVNet("/home/ierturk/Work/REPOs/yolo/yoloIE/weights/converted.weights",
                   "/home/ierturk/Work/REPOs/ml/data/ssdData/vott/vott-yolo-export/yolov3-tiny.cfg",
                   "darknet",
                   "/home/ierturk/Work/REPOs/ml/data/ssdData/vott/vott-yolo-export/class.names")

    def run(self):
        self.capture.process = True
        self.capture.start()

        while self.process:
            frame = None
            try:
                frame = self.capture.queue.get_nowait()
                self.capture.queue.queue.clear()

            except queue.Empty:
                pass

            if not frame is None:
                self.net.set_input(frame)
                self.net.forward()
                netIOs = self.net.get_output()
                netIOs.processedFrame = cv.resize(netIOs.originalFrame, (512, 288))
                self.post_process(netIOs)
                self.queue.put(netIOs.processedFrame)

        self.capture.process = False
        self.capture.join()

    def post_process(self, netIOs):
        frame_height = netIOs.processedFrame.shape[0]
        frame_width = netIOs.processedFrame.shape[1]

        class_ids = []
        confidences = []
        boxes = []
        for out in netIOs.output:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.net.confThreshold:
                    center_x = int(detection[0] * frame_width)
                    center_y = int((detection[1] - 70/320) * frame_height * 320/180)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height * 320/180)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    class_ids.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indices = cv.dnn.NMSBoxes(boxes, confidences, self.net.confThreshold, self.net.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.draw_pred(netIOs, class_ids[i], confidences[i], left, top, left + width, top + height)

        if self.queue.counter > 1:
            label = 'Camera: %.2f FPS' % (self.capture.queue.getFPS())
            cv.putText(netIOs.processedFrame, label, (320, 250), cv.FONT_HERSHEY_SIMPLEX, 0.5, (201, 161, 51))

            label = 'Network: %.2f FPS' % (self.queue.getFPS())
            cv.putText(netIOs.processedFrame, label, (320, 265), cv.FONT_HERSHEY_SIMPLEX, 0.5, (201, 161, 51))

            label = 'Skipped: %d' % (self.capture.queue.counter - self.queue.counter)
            cv.putText(netIOs.processedFrame, label, (320, 280), cv.FONT_HERSHEY_SIMPLEX, 0.5, (201, 161, 51))

    def draw_pred(self, netIOs, class_id, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv.rectangle(netIOs.processedFrame, (left, top), (right, bottom), (0, 255, 0))

        label = '%.2f' % conf

        # Print a label of class.
        if self.net.classes:
            assert (class_id < len(self.net.classes))
            label = '%s: %s' % (self.net.classes[class_id], label)

        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(netIOs.processedFrame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine),
                     (255, 255, 255),
                     cv.FILLED)
        cv.putText(netIOs.processedFrame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
