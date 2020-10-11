#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 21:27:23 2020

template is taken from:
    https://www.learnopencv.com/training-a-custom-object-detector-with-dlib-making-gesture-controlled-applications/?ck_subscriber_id=546165186
@author: axeh
"""
import tensorflow as tf
import cv2 as cv
import os
import numpy as np
import argparse
import time

from utils.perlin_flow import PerlinFlow
from utils.pattern_utils import Pattern, Polygon
from utils.pattern_utils import FistPatternEffect, HandPatternEffect, JutsuPatternEffect

#%% parse input data
parser = argparse.ArgumentParser()
parser.add_argument("--detectordir", 
                    default=None)
args = vars(parser.parse_args())

if args["detectordir"] is None:
    DETECTOR_DIR = "./dnn/ssd_mobilenet_gesture_detector/"
else:
    DETECTOR_DIR = args["detectordir"]

#%% load detectors
tf.keras.backend.clear_session()
detect_fn = tf.saved_model.load(DETECTOR_DIR)
colors = [(np.random.randint(0,255),np.random.randint(0,255), np.random.randint(0,255))
          for _ in range(3)]
classes = ["hand", "fist", "teleportation_jutsu"]

#%% define class effects
fistpattern = FistPatternEffect()
handpattern = HandPatternEffect()
jutsupattern = JutsuPatternEffect()
        

#%% start video capture
cv.namedWindow("frame")
w,h = 640, 480

# initialize web cam
cap = cv.VideoCapture(0)#, cv.CAP_DSHOW)

# initially the size of the hand and its center's x coor will be at 0
#(MR: we're using sliding windows...)
size, center_x = 0,0

st = time.time()

# start video capture
while True:
    ret,frame = cap.read()
    
    
    if not ret:
        break
    
    # flip the frame
    frame = cv.resize(frame, (w,h))
    frame = cv.flip(frame, 1)
    
    # create a copy of the frame
    copy = frame.copy()

    # convert to right format
    input_tensor = np.expand_dims(frame, axis=0)
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.uint8)
    
    # get detections
    detections = detect_fn(input_tensor)
    
    jutsu_detected = False
    jutsu_pt1, jutsu_pt2 = (None,None), (None,None)
    for box,clss,score in zip(detections["detection_boxes"][0], # 0-dim - batch
                              detections["detection_classes"][0],
                              detections["detection_scores"][0]):
        box = box.numpy()
        clss = clss.numpy().astype(np.uint32)# + label_id_offset
        score = score.numpy()

        # detections are sorted based on scores in descending order
        if score < 0.5:
            break

        # get box coordinates
        y1,x1,y2,x2 = (box*np.array([h,w,h,w])).astype(int)
        
        # Recall: class indices start from `1` (`0` is reserved for background)
        if classes[clss-1] == "hand":
            handpattern.draw_pattern(frame, (x1,y1), (x2,y2))
        elif classes[clss-1] == "fist":
            fistpattern.draw_pattern(frame, (x1,y1), (x2,y2))            
        elif classes[clss-1] == "teleportation_jutsu":
            jutsu_detected = True
            justu_pt1, jutsu_pt2 = (x1,y1), (x2,y2)    
    jutsupattern.draw_pattern(frame, jutsu_detected, jutsu_pt1, jutsu_pt2)
        
    # display 
    cv.imshow("frame", frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()
        