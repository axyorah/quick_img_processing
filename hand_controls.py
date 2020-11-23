#!/usr/bin/env python3
"""
Created on Tue Aug 20 10:06:39 2019

@author: axeh
"""

import numpy as np
import tensorflow as tf
import cv2 as cv
import imutils
from imutils.video import VideoStream
import argparse
import time
from utils.hand_utils import add_buttons, get_button_masks, blur_box

#%% ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--show_hand_mask", default="0",
                    help="set to 1 to show hand mask" +\
                         "(as filled bounding boxes on a separate frame)" +\
                         "and to 0 to ignore it (default)")
parser.add_argument("-b", "--show_hand_bbox", default="0",
                    help="set to 1 to show hand bounding boxes" +\
                         "and to 0 to ignore it (default)")

args = vars(parser.parse_args())

#%% ---------------------------------------------------------------------------
# realtive paths to models
PATH_TO_FACE_FILTER  = "./dnn/haarcascade_frontalface_default.xml"
DETECTOR_DIR = "./dnn/efficientdet_hand_detector/"

#%% ---------------------------------------------------------------------------
# get out-of-the-box face filter from opencv
face_cascade = cv.CascadeClassifier(PATH_TO_FACE_FILTER)

# load the inference model for hand detector
tf.keras.backend.clear_session()
detect_fn = tf.saved_model.load(DETECTOR_DIR)
classes = ["hand"]

#%% ---------------------------------------------------------------------------
# useful function
def get_hand_masks(detections, frame, threshold=0.5,
                   show_bbox=0, show_mask=0):
    h,w = frame.shape[:2]
    handmask = np.zeros(frame.shape)
    for box,clss,score in zip(detections["detection_boxes"][0], # 0-dim - batch
                              detections["detection_classes"][0],
                              detections["detection_scores"][0]):
        box = box.numpy()
        clss = clss.numpy().astype(np.uint32) # 1,2,3... (0 is reserved for background)
        score = score.numpy()

        # detections are sorted based on object probability scores in descending order;
        # stop when score drops below threshold 
        if score < threshold:
            break

        # get box coordinates
        y1,x1,y2,x2 = (box*np.array([h,w,h,w])).astype(int)
        handmask[y1:y2,x1:x2,:] = 255

        # show additional info if requested
        if show_bbox:
            cv.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)

    if show_mask:
        cv.imshow("handmask", handmask)

    return handmask

#%% ---------------------------------------------------------------------------
# initiate video stream
vs = VideoStream(src=0).start()
time.sleep(2)

# initiate parameters controled via `buttons`
width = 640
blur  = 0

# sample video stream to get some params
sample = vs.read()
sample = imutils.resize(sample, width=width)
h,w = sample.shape[:2]
aspect_ratio = h/w

# get some ref values
h_pre,w_pre = 0,0

while True:
    # read video frame
    frame = vs.read()
    
    # flip the frame
    frame = cv.resize(frame, (width, int(width * aspect_ratio)))
    frame = cv.flip(frame, 1)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # convert to right format
    input_tensor = np.expand_dims(frame, axis=0)
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.uint8)
    
    # get detections for current frame
    #(will always make 100 detections sorted by object probability score)
    detections = detect_fn(input_tensor)

    # ----------------------------------------------------------------- 
    # run inference to get hand masks
    handmask = get_hand_masks(
        detections, frame, threshold=0.7,
        show_bbox=int(args["show_hand_bbox"]),
        show_mask=int(args["show_hand_mask"]))

    # -----------------------------------------------------------------
    # draw buttons
    frame, buttons = add_buttons(frame)

    # resample the button masks if frame dimensions have changed
    if (h_pre,w_pre) != (h,w):
        masks = get_button_masks(frame, buttons)

    # -----------------------------------------------------------------    
    # check if any of the buttons overlaps with a hand
    for name in buttons.keys():
        overlap  = cv.bitwise_and(handmask,handmask,mask=masks[name])

        if np.sum(overlap) / np.sum(masks[name]) > 0.75:
            width, blur = buttons[name].action([width, blur])

    # -----------------------------------------------------------------
    # update vars affected by button controls
    width = np.clip(width, 200, 1000)
    blur  = np.clip(blur, 0, 20)
    k     = 1 + 2*blur
    sigma = 1 +   blur
            
    # get face bbox and update face blur
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        frame = blur_box(frame, (x,y), (x+w,y+h), k, sigma)

    # -----------------------------------------------------------------
    # display final frame
    cv.imshow("press 'q' to quit",  frame[:,:,::-1])
    h_pre,w_pre = frame.shape[:2]
    stopkey = cv.waitKey(1)
    if stopkey == ord("q"):
        break

cv.destroyAllWindows()
vs.stop()   