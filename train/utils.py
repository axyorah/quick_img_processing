import cv2 as cv
import time

class SlidingWindow:
    def __init__(self, w=130, h=190, dw=10, dh=10, skip=17, init_count=0):
        self.w = w
        self.h = h
        self.dw = dw
        self.dh = dh
        self.skip = skip # num of frames to skip before updating the window position

        # "current" sliding window position
        self.counter = init_count
        self.x1 = 0
        self.y1 = 10
        self.irow = 0

    def update_sliding_window_position(self, frame):
        # find position of the sliding window
        if not self.counter % self.skip:
            # adjust x1,y1
            # for even rows slide window to the right
            # for odd rows slide window to the left
            if not self.irow % 2:            
                if self.x1 + self.w < frame.shape[1]:
                    self.x1 += self.dw
                    time.sleep(0.1)
                # if sliding window has reached the end of the row - 
                # start the new row at the same x1 (next row will slide to the left)
                elif self.y1 + self.h < frame.shape[0]:
                    self.y1 += self.dh  
                    self.irow += 1
                else:
                    self.x1 = 0
                    self.y1 = 10
            else:
                if self.x1 > 0:
                    self.x1 -= self.dw
                    time.sleep(0.1)
                # if sliding window has reached the end of the row - 
                # start the new row at x1 = 0
                elif self.y1 + self.h < frame.shape[0]:
                    self.y1 += self.dh
                    self.x1 = 0
                    self.irow += 1
                else:
                    self.x1 = 0
                    self.y1 = 10

        self.counter += 1

    def add_sliding_window_to_frame(self, frame):
        cv.rectangle(
            frame, 
            (self.x1,self.y1), 
            (self.x1+self.w, self.y1+self.h), 
            (0,255,0), 
            3
        )

    def write_sliding_window_position_to_file(self, filename):
        with open(filename, "a") as f:
            f.write(f"{self.counter}:{self.x1},{self.y1},{self.x1+self.w},{self.y1+self.h}\n")
