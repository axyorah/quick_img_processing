#!/usr/bin/env python3
from typing import List, Tuple, Set, Dict, Optional, Union

import torch
import cv2 as cv
import numpy as np
import time

from utils.effect_utils import (
    HaSEffect,
    SpellPatternEffect,
    KaboomPatternEffect, 
    LightningPatternEffect
)

from utils.detector_utils import YoloTorchDetector

DETECTOR_DICT = "./dnn/yolov5_gesture_state_dict.pt"
DETECTOR_YAML = "./dnn/yolov5s.yaml"
IMGSZ = 448
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
HALF = False


def adjust_frame(frame, tar_sz):
    frame = cv.resize(frame, tar_sz)
    frame = cv.flip(frame, 1)
    return frame

def draw_effects(frame, detections, classes):
    h,w = frame.shape[:2]

    jutsu_detected, lightning_detected = False, False
    jutsu_pt1, jutsu_pt2 = (None,None), (None,None) #TODO: make explosion appear from the center of the box
    lightning_pt1, lightning_pt2 = (None,None), (None,None)
    
    for x1,y1,x2,y2,prob,clss in detections:

        # Recall: class indices start from 1 (0 is reserved for background)
        if classes[clss] == "hand":
            spell.draw_pattern(frame, (x1,y1), (x2,y2))
        
        elif classes[clss] == "fist":
            has.next()
            has.translate((x1,y1), (x2,y2))
            has.scale((x1,y1), (x2,y2))
            has.draw(frame)

            #has.draw_pattern(frame, (x1,y1), (x2,y2))            
        
        elif classes[clss] == "teleportation_jutsu":
            # we can't afford many false-positives for teleportation_jutsu
            # as each detection would trigger 20-frames-long uninterruptible animation;
            # so let's store bollean `jutsu_detected` over the last 10 frames
            # and show the animation only if 5/10 frames had `jutsu_detected=True`
            #(this is resolved in JutsuPatternEffect.draw_pattern())
            jutsu_detected = True
            jutsu_pt1, jutsu_pt2 = (x1,y1), (x2,y2)
        
        elif classes[clss]  == "horns":
            lightning_detected = True
            lightning_pt1, lightning_pt2 = (x1,y1), (x2,y2)
    
    kaboom.draw_pattern(
        frame, jutsu_detected, jutsu_pt1, jutsu_pt2
    )

    lightning.draw_pattern(
        frame, lightning_detected, lightning_pt1, lightning_pt2
    )


def main():
    # adjust float precision: if no cuda - alawys use float32 instead of float16/float32
    half = False if DEVICE.type == 'cpu' else HALF

    classes = [
        'fist', 'hand', 'horns', 'teleportation_jutsu', 'tori_sign'
    ]

    # load model
    detector = YoloTorchDetector(
        DETECTOR_YAML, 
        DETECTOR_DICT, 
        class_dict = {
            i: name for i, name in enumerate(classes, start=1)
        },
        device=DEVICE, 
        half=half
    )
    

    # start video capture
    cv.namedWindow("frame")
    w,h = 640, 480

    # initialize web cam
    vc = cv.VideoCapture(0)
    time.sleep(2)

    # start video capture
    while True:
        # read video frame
        _, frame = vc.read()
    
        # flip the frame
        frame = adjust_frame(frame, (w,h))

        # convert to right format
        input_tensor = detector.preprocess(frame)
    
        # get detections for current frame
        detections = detector.detect(
            input_tensor, 
            threshold=0.75,
            sz=frame.shape[:2]
        )

        # check detections and draw corresponding effects
        draw_effects(frame, detections, classes)        
        
        # display 
        cv.imshow("press `q` to quit", frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break
    
    vc.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    # define class effects
    has = HaSEffect()
    spell = SpellPatternEffect()
    kaboom = KaboomPatternEffect()   
    lightning = LightningPatternEffect()  

    main()