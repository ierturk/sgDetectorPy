import queue
import cv2 as cv
import numpy as np
from process import ProcessChannel


class Detector(object):
    def __init__(self):
        super(Detector, self).__init__()

        self.winName = 'sgDetector'
        cv.namedWindow(self.winName, cv.WINDOW_NORMAL)

        self.process_channel_right = ProcessChannel("http://localhost:8080/ccdff8d9bcfe4f8524bf36810019b860/mp4/pnOOUNJfOo/RightCam/s.mp4")
        self.process_channel_left = ProcessChannel("http://localhost:8080/ccdff8d9bcfe4f8524bf36810019b860/mp4/pnOOUNJfOo/LeftCam/s.mp4")

    def run(self):
        while cv.waitKey(1) < 0:
            try:

                frameR = self.process_channel_right.queue.get_nowait()
                frameL = self.process_channel_left.queue.get_nowait()
                image = np.concatenate((frameR, frameL), axis=0)
                cv.imshow(self.winName, image)

            except Exception as e:
                pass


def main():
    dt = Detector()
    dt.process_channel_right.process = True
    dt.process_channel_left.process = True
    dt.process_channel_right.start()
    dt.process_channel_left.start()

    dt.run()

    cv.destroyAllWindows()
    dt.process_channel_right.process = False
    dt.process_channel_left.process = False
    dt.process_channel_right.join()
    dt.process_channel_left.join()


if __name__ == '__main__':
    main()
