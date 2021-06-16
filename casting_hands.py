#!/usr/bin/env python3
import tensorflow as tf
import cv2 as cv
import numpy as np
import time

from utils.pattern_utils import \
    FistPatternEffect, HandPatternEffect, \
    JutsuPatternEffect, LightningPatternEffect

DETECTOR_DIR = "./dnn/ssd_mobilenet_gesture_detector/"

def adjust_frame(frame, tar_sz):
    frame = cv.resize(frame, tar_sz)
    frame = cv.flip(frame, 1)
    return frame

def preprocess_frame(frame):
    input_tensor = np.expand_dims(frame, axis=0)
    return tf.convert_to_tensor(input_tensor, dtype=tf.uint8)

def draw_effects(frame, detections):
    h,w = frame.shape[:2]

    jutsu_detected, lightning_detected = False, False
    jutsu_pt1, jutsu_pt2 = (None,None), (None,None) #TODO: make explosion appear from the center of the box
    lightning_pt1, lightning_pt2 = (None,None), (None,None)
    
    for box,clss,score in zip(detections["detection_boxes"][0], # 0-dim - batch
                              detections["detection_classes"][0],
                              detections["detection_scores"][0]):
        box = box.numpy()
        clss = clss.numpy().astype(np.uint32) # 1,2,3... (0 is reserved for background)
        score = score.numpy()

        # detections are sorted based on object probability scores in descending order;
        # stop when score drops below threshold (0.5)
        if score < 0.5:
            break

        # get box coordinates
        y1,x1,y2,x2 = (box*np.array([h,w,h,w])).astype(int)
        
        # Recall: class indices start from 1 (0 is reserved for background)
        if classes[clss-1] == "hand":
            handpattern.draw_pattern(frame, (x1,y1), (x2,y2))
        elif classes[clss-1] == "fist":
            fistpattern.draw_pattern(frame, (x1,y1), (x2,y2))            
        elif classes[clss-1] == "teleportation_jutsu":
            # we can't afford many false-positives for teleportation_jutsu
            # as each detection would trigger 20-frames-long uninterruptible animation;
            # so let's store bollean `jutsu_detected` over the last 10 frames
            # and show the animation only if 5/10 frames had `jutsu_detected=True`
            #(this is resolved in JutsuPatternEffect.draw_pattern())
            jutsu_detected = True
            jutsu_pt1, jutsu_pt2 = (x1,y1), (x2,y2)
        elif classes[clss-1]  == "horns":
            lightning_detected = True
            lightning_pt1, lightning_pt2 = (x1,y1), (x2,y2)
    jutsupattern.draw_pattern(
        frame, jutsu_detected, jutsu_pt1, jutsu_pt2)
    lightningpattern.draw_pattern(
        frame, lightning_detected, lightning_pt1, lightning_pt2)


def main():
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
        input_tensor = preprocess_frame(frame)
    
        # get detections for current frame
        #(will always make 100 detections sorted by object probability score)
        detections = detect_fn(input_tensor)

        # check detections and draw corresponding effects
        draw_effects(frame, detections)        
        
        # display 
        cv.imshow("press `q` to quit", frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break
    
    vc.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    # load hand gesture qdetector
    tf.keras.backend.clear_session()
    detect_fn = tf.saved_model.load(DETECTOR_DIR)
    classes = ["hand", "fist", "teleportation_jutsu", "tori_sign", "horns"]

    # define class effects
    fistpattern = FistPatternEffect()
    handpattern = HandPatternEffect()
    jutsupattern = JutsuPatternEffect()   
    lightningpattern = LightningPatternEffect()  

    main()