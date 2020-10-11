# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 21:27:23 2020
on Windows: `facial` venv

tutorial is taken from:
    https://www.learnopencv.com/training-a-custom-object-detector-with-dlib-making-gesture-controlled-applications/?ck_subscriber_id=546165186
@author: axeh
"""

import dlib
import cv2 as cv
#import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
#import pyautogui as pyg
import shutil
import argparse
import time

from utils.pattern_utils import Pattern, Polygon
from utils.perlin_flow import PerlinFlow
from utils.hand_utils import FistPatternEffect, HandPatternEffect, JutsuPatternEffect

#%% parse input data
parser = argparse.ArgumentParser()
parser.add_argument("--detectorpaths", 
                    default=None)
args = vars(parser.parse_args())

if args["detectorpaths"] is None:
    rootdir = "D:/code_stuff/python_stuff/courses/computer_vision/dnn/" \
    if "win" in sys.platform else \
    "/media/axeh/code_stuff/python_stuff/courses/computer_vision/dnn/"
    
    detectorpaths = [os.path.join(rootdir,f"hog_{dclass}_detector.svm")
                     for dclass in ["hand", "fist", "teleportation_jutsu"]]    
else:
    detectorpaths = args["detectorpaths"].split(" ")

#%% load detectors
detectors = [dlib.fhog_object_detector(detectorpath)
             for detectorpath in detectorpaths]
colors = [(np.random.randint(0,255),np.random.randint(0,255), np.random.randint(0,255))
          for _ in range(len(detectors))]
classes = ["_".join(detectorpath.replace(os.path.sep, "/")\
                                .split("/")[-1].split("_")[1:-1]) 
           for detectorpath in detectorpaths]

#%% define class effects

            
fistpattern = FistPatternEffect()
handpattern = HandPatternEffect()
jutsupattern = JutsuPatternEffect()
        

#%% start video capture
cv.namedWindow("frame")

# initialize web cam
cap = cv.VideoCapture(0, cv.CAP_DSHOW)

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
    frame = cv.flip(frame, 1)
    
    # create a copy of the frame
    copy = frame.copy()
    
    # get detections
    [detections, confidences, indices] = \
        dlib.fhog_object_detector.run_multiple(
                detectors, frame, upsample_num_times=1)
    
    jutsu_detected = False
    for detection,confidence,idx in zip(detections, confidences, indices):
        idx = int(idx)
        #if confidence < 0.5:
        #    continue
        
        x1 = int(detection.left())
        y1 = int(detection.top())
        x2 = int(detection.right())
        y2 = int(detection.bottom())
        
        #cv.rectangle(frame, (x1,y1), (x2,y2), colors[int(idx)], 3)
        if classes[idx] == "hand":
            #show_hand_effect(frame, (x1,y1), (x2,y2))
            handpattern.draw_pattern(frame, (x1,y1), (x2,y2))
        elif classes[idx] == "fist":
            fistpattern.draw_pattern(frame, (x1,y1), (x2,y2))            
        elif classes[idx] == "teleportation_jutsu":
            jutsu_detected = True
            justu_pt1, jutsu_pt2 = (x1,y1), (x2,y2)    
    jutsupattern.draw_pattern(frame, jutsu_detected)
        
    # display 
    cv.imshow("frame", frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()
        