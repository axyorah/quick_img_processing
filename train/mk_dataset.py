# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 16:36:33 2020

most code is taken from:
    https://www.learnopencv.com/training-a-custom-object-detector-with-dlib-making-gesture-controlled-applications/?ck_subscriber_id=546165186
"""

import cv2 as cv
import os
import shutil
import argparse
from utils import BBoxWriter

def get_args():
    """parse input data"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasetroot", 
        default="dataset",
        help="desired path for storing the records\n"+\
            "by default will create directory `dataset` in current directory"
    )
    parser.add_argument(
        "--class", 
        default="hand",
        help="recorded class; (default  is `hand`)"
    )
    parser.add_argument(
        "--clear", 
        default="False",
        help="flag indicating whether all old records for the chosen class need to be overwritten\n"+\
            "(default `False`, set it to `True` if you want to overwrite)"
    )
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

def prepare_dirs(root, clss, cleanup=False):
    """create or update image directory and text file with bbox records"""
    subdir = os.path.join(root, clss)
    box_file = os.path.join(root, "boxes_"+clss+".txt")

    print(f"[INFO] Creating a dataset for class `{clss}`")
    print(f"       Recorded frames will be stored in `{subdir}`")
    print(f"       Bounding box info will be stored in `{box_file}`")

    # double check if we really want to overwrite the previous records
    if cleanup == True:
        ans = input(
            f"[INFO] Are you sure you want to clear"+\
            f"the previous contents of\n"+\
            f"      `{subdir}` and `{box_file}`?(y/n)\n")
        if ans == "y" or ans == "Y":
            cleanup = True
            print("[INFO] The previous records will be removed.")
        else:
            cleanup = False
            print(f"[INFO] New records will be appended to the old records at `{subdir}`.")
    else:
        print("[INFO] New records will be appended to the old records at `{subdir}`.")


    # prepare dirs and files for writing the records
    #counter = 0 # count already exising records (if they don't need to be cleaned up)
    if cleanup:
        # delete the img dir if it exists
        if os.path.exists(subdir):
            print(f"[INFO] Deleting `{subdir}`.")
            shutil.rmtree(subdir)
        
        # clear up all previous bounding boxes
        if os.path.exists(box_file):
            print(f"[INFO] Deleting `{box_file}`.")
            with open(box_file, "w") as f:
                f.write("")
    
    # create img dir if it doesn't exist
    if not os.path.exists(root):
        print(f"[INFO] Creating `{root}`.")
        os.mkdir(root)
    if not os.path.exists(subdir):
        print(f"[INFO] Creating `{subdir}`.")
        os.mkdir(subdir)

    # open box_file or create it if it doesn't exist
    if not os.path.exists(box_file):
        print(f"[INFO] Creating `{box_file}`.")
        with open(box_file, "w+") as f:
            pass

    #return counter
    
def add_init_text_to_frame(frame, text):
    h, w = frame.shape[:2]
    xcenter, ycenter = w // 2, h // 2

    fontScale = 2
    fontFace = cv.FONT_HERSHEY_SIMPLEX
    pos = (xcenter - 15*fontScale*len(text)//2, ycenter)

    cv.putText(frame, text, pos, fontFace, fontScale, (0,255,0), 2)

def adjust_frame(frame, tar_sz):
    frame = cv.resize(frame, tar_sz)
    frame = cv.flip(frame, 1)
    return frame

def main():
    # get args
    args = get_args()
    
    root = args["datasetroot"]
    clss = args["class"]
    cleanup = True if args["clear"] == "True" else False
    skip_frame = int(args["skip"])    
    win_w, win_h = map(int, args["winsize"].split("x"))

    subdir = os.path.join(root, clss)
    box_file = os.path.join(root, "boxes_"+clss+".txt")
    prepare_dirs(root, clss, cleanup)
    
    # initiate sliding window drawer
    print(f"[INFO] Setting the size of a sliding window to {win_w}x{win_h}")
    writer = BBoxWriter(win_w, win_h, skip=skip_frame, img_dir=subdir, bbox_file=box_file)    
    
    # prepare frame
    w, h = 640, 480 # frame dimensions
    window_name = f"Recording class `{args['class']}` (press `q` to quit)"
    cv.namedWindow(window_name)

    # initialize webcam
    print("[INFO] Starting Video Recording...")
    cap = cv.VideoCapture(0)#, cv.CAP_DSHOW)

    # dont record the initial 60 frame, so that there's time to adjust the hand
    init_wait, init_wait_counter = 60, 0

    while True:
        ret,frame = cap.read()
        if not ret:
            break
    
        # flip/resize the 
        frame = adjust_frame(frame, (w,h))
    
        if init_wait_counter == 0:
            print(f"[INFO] Frame size: {frame.shape[1]}x{frame.shape[0]}")
        
        # skip first 60 frames until the hand is placed correctly
        # after that - adjust the location of the sliding window and make records 
        if init_wait_counter < init_wait:             
            add_init_text_to_frame(frame, clss)
            init_wait_counter +=1        
        else:
            # call function to bbox position on each frame
            # the actual updates and writes will be done every `skip` number of frames
            writer.update_bbox_position(frame, write=True)
        
        writer.add_bbox_to_frame(frame)
    
        # show frame
        cv.imshow(window_name, frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()    


if __name__ == "__main__":
    main()