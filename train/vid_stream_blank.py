import cv2 as cv
import time
import argparse
from utils import SlidingWindow

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--winsize", 
        default="130x190",
        help="size of the sliding window in pixels: `width`x`height` (default 130x190)"
    )
    parser.add_argument(
        "--skip", 
        default="17",
        help="number of frames to be skipped between the recordings (dafault 17);\n"+\
            "can be used to regulate the speed of the sliding window: \n"+\
            "increment `skip` to speed up the sliding window and \n"+\
            "decrement it to slow the sliding window down"
    )

    return vars(parser.parse_args())



def adjust_frame(frame, tar_sz):
    frame = cv.resize(frame, tar_sz)
    frame = cv.flip(frame, 1)
    return frame

def main():
    # get args
    args = get_args()
    skip_frame = int(args["skip"])
    winsize = map(int, args["winsize"].split("x"))
    win_w, win_h = winsize

    # initiate the sliding window drawer
    drawer = SlidingWindow(win_w, win_h, skip_frame)

    # start video capture
    name = "press `q` to quit"
    cv.namedWindow(name)
    w,h = 640, 480

    # initialize web cam
    vc = cv.VideoCapture(0)
    time.sleep(2)

    # start video capture
    while True:
        # read video frame
        _, frame = vc.read()
    
        # flip/resize the frame
        frame = adjust_frame(frame, (w,h))

        # add sliding window to frame
        drawer.update_sliding_window_position(frame)
        drawer.add_sliding_window_to_frame(frame)
        
        # display 
        cv.imshow(name, frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break
    
    vc.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()