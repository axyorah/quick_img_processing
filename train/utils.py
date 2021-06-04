import cv2 as cv
import time
import os

class BBoxWriter:
    def __init__(
        self, w=130, h=190, dw=10, dh=10, skip=17, init_count=0, 
        img_dir='dataset/hand', bbox_file='dataset/boxes_hand.txt'
        ):
        # paths for saving
        self.img_dir = img_dir
        self.bbox_file = bbox_file

        # bbox specs
        self.w = w # bbox width
        self.h = h # bbox height
        self.dw = dw # bbox stride along hor direction
        self.dh = dh # bbox stride along vert direction
        self.skip = skip # num of frames to skip before updating the bbox position

        # "current" sliding window position
        self.write_counter = init_count # only counts frames that are written
        self.counter = 0 # counts every frame
        self.x1 = 0
        self.y1 = 10
        self.irow = 0
        

    def update_sliding_window_position(self, frame, write=False):
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

            if write:
                self.write_frame_and_sliding_window_position(frame)

        self.counter += 1

    def add_sliding_window_to_frame(self, frame):
        cv.rectangle(
            frame, 
            (self.x1,self.y1), 
            (self.x1+self.w, self.y1+self.h), 
            (0,255,0), 
            3
        )

    def write_frame_and_sliding_window_position(self, frame):
        self.create_imgdir_and_bboxfile_if_needed()

        # write frame
        frame_name = os.path.join(self.img_dir, f"{self.write_counter}.jpeg")
        cv.imwrite(frame_name, frame)

        # write position
        with open(self.bbox_file, "a") as f:
            f.write(f"{self.write_counter}:{self.x1},{self.y1},{self.x1+self.w},{self.y1+self.h}\n")
        self.write_counter += 1

    def create_imgdir_and_bboxfile_if_needed(self):
        if (not os.path.exists(self.img_dir)) or (not os.path.exists(self.bbox_file)):
            # reset write counter
            self.write_counter = 0

            # create dir to store frames
            os.makedirs(self.img_dir)

            # create file to store bbox records
            filepath = self.bbox_file.split(os.path.sep)
            os.makedirs(os.path.join(*filepath[:-1]), exist_ok=True)
            with open(self.bbox_file, "w") as f:
                f.write("")
