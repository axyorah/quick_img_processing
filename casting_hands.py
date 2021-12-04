#!/usr/bin/env python3
from typing import List, Tuple, Set, Dict, Optional, Union

import torch
import cv2 as cv
import numpy as np
import time

from abc import ABC

from utils.detector_utils import YoloTorchDetector
from utils.hand_utils import Event, Observable

from utils.effect_utils import (
    HaSEffect,
    SpellEffect,
    KaboomEffect,
    LightningEffect
)


DETECTOR_DICT = "./dnn/yolov5_gesture_state_dict.pt"
DETECTOR_YAML = "./dnn/yolov5s.yaml"
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
HALF = False


class Announcer(Observable):
    def __init__(self):
        self.detection = Event()
        self.resolution = Event()

    def announce(self, frame, detections):
        for x1,y1,x2,y2,prob,clss in detections:
            self.detection(frame, clss, (x1,y1), (x2,y2))

    def resolve(self, frame):
        self.resolution(frame)


class Triggerer(ABC):
    def __init__(self, clss, effector, announcer):
        self.clss = clss
        self.effector = effector
        self.announcer = announcer
        self.announcer.detection.append(self.register)
        self.announcer.resolution.append(self.resolve)

    def register(self, frame, clss, pt1, pt2):
        """
        will be called for each detection 
        with class probability above threshold 
        """
        pass

    def resolve(self, frame):
        """
        will be called once for each frame
        after all detections have been announced
        """
        pass


class PerlinTriggerer(Triggerer):
    def __init__(self, clss, effector, announcer):
        super().__init__(clss, effector, announcer)

    def register(self, frame, clss, pt1, pt2):
        # we want to draw the effect as soon as it is detected
        if clss == self.clss:
            self.effector.translate(pt1, pt2)
            self.effector.scale(pt1, pt2)
            self.effector.draw(frame)

    def resolve(self, frame):
        self.effector.next()

class SequenceTriggerer(Triggerer):
    """
    frame sequence is long and expensive,
    so we want to minimize false pos as much as possible;
    therefore, sequence effect will only be triggered
    if more than a `que_threshold` frames of the 
    last `que_len` frames had some trigger detected
    """
    def __init__(self, clss, effector, announcer, que_len=10, que_threshold=5):
        super().__init__(clss, effector, announcer)

        self.que = [False] * que_len
        self.que_threshold = que_threshold
        self.curr_detected = False

        self.pt1 = (None,None)
        self.pt2 = (None,None)

    def register(self, frame, clss, pt1, pt2):
        """
        we only want to check if class was detected
        anywhere on the current frame, which is the same as
        any of the detections corresponds to `this` class;
        we just record it, but don't do anything about it yet
        """
        if clss == self.clss and not self.curr_detected:
            self.curr_detected = True
            self.pt1 = pt1
            self.pt2 = pt2

    def resolve(self, frame):
        """
        maybe start a new sequence or continue the ongoing one
        """
        # always update que
        self.que.pop(0)
        self.que.append(self.curr_detected)

        # maybe draw if detected or animation is ongoing
        if sum(self.que) >= self.que_threshold or self.effector.isongoing:
            self.effector.maybe_begin()
            self.effector.maybe_draw(frame, self.pt1, self.pt2)

        # reset
        self.curr_detected = False
        self.pt1 = (None,None)
        self.pt2 = (None,None)


def adjust_frame(frame, tar_sz):
    frame = cv.resize(frame, tar_sz)
    frame = cv.flip(frame, 1)
    return frame


def main():
    # adjust float precision: if no cuda - alawys use float32 instead of float16/float32
    half = False if DEVICE.type == 'cpu' else HALF

    classes = [
        'fist', 'hand', 'horns', 'teleportation_jutsu', 'tori_sign'
    ]

    # load detector
    detector = YoloTorchDetector(
        DETECTOR_YAML, 
        DETECTOR_DICT, 
        class_dict = {
            i: name for i, name in enumerate(classes)
        },
        device=DEVICE, 
        half=half
    )


    # get effect triggerers that observer `Announcer``
    announcer = Announcer()

    PerlinTriggerer(0, HaSEffect(), announcer)
    PerlinTriggerer(1, SpellEffect(), announcer)
    SequenceTriggerer(2, LightningEffect(), announcer)
    SequenceTriggerer(3, KaboomEffect(), announcer)


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
            threshold=0.6,
            sz=frame.shape[:2]
        )

        # check detections and draw corresponding effects
        announcer.announce(frame, detections)
        announcer.resolve(frame)
        
        # display 
        cv.imshow("press `q` to quit", frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break
    
    vc.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()