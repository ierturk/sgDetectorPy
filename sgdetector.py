import cv2 as cv
import numpy as np
from process import ProcessChannel


class Detector(object):
    def __init__(self):
        super(Detector, self).__init__()

        self.winName = 'sgDetector'
        cv.namedWindow(self.winName, cv.WINDOW_NORMAL)
        cv.createTrackbar('pos', self.winName, 0, 19999, self.callbackTrackBar)
        self.process_channel_right = ProcessChannel("/home/ierturk/Work/REPOs/ml/data/sgDetector/Video/2019-10-10T18-15-01-right.mp4")
        self.process_channel_left = ProcessChannel("/home/ierturk/Work/REPOs/ml/data/sgDetector/Video/2019-10-10T18-15-02-left.mp4")
        self.stat = False

    def callbackTrackBar(self, pos):
        # print(pos)
        if self.stat:
            self.process_channel_right.capture.current_frame = \
                int(pos * self.process_channel_right.capture.frame_count / 20000)
            self.process_channel_left.capture.current_frame = \
                int(pos * self.process_channel_left.capture.frame_count / 20000)

    def run(self):
        while True:
            ch = cv.waitKeyEx(1)
            if ch == 27:
                break
            elif ch == 32:
                # print('space bar')
                self.stat = not self.stat
                self.process_channel_right.capture.stat = self.stat
                self.process_channel_left.capture.stat = self.stat

            if not self.stat:
                cv.setTrackbarPos('pos',
                                  'sgDetector',
                                  int(19999 * self.process_channel_right.capture.current_frame
                                      / self.process_channel_right.capture.frame_count))

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
