#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import cv2 as cv
import argparse
import time
from utils.hand_utils import mk_buttons, add_buttons, blur_box

FACE_FILTER_PATH  = "./dnn/haarcascade_frontalface_default.xml"
DETECTOR_DIR = "./dnn/efficientdet_hand_detector/"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--show_hand_mask", default="0",
        help="set to 1 to show hand mask" +\
            "(as filled bounding boxes on a separate frame)" +\
            "and to 0 to ignore it (default)")
    parser.add_argument(
        "-b", "--show_hand_bbox", default="0",
        help="set to 1 to show hand bounding boxes" +\
            "and to 0 to ignore it (default)")

    args = vars(parser.parse_args())

    return args

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

def adjust_frame(frame, tar_sz):
    frame = cv.resize(frame, tar_sz)
    frame = cv.flip(frame, 1)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    return frame

def preprocess_frame(frame):
    input_tensor = np.expand_dims(frame, axis=0)
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.uint8)
    return input_tensor

def get_aspect_ratio(frame):
    h,w = frame.shape[:2]
    return h/w

def update_params(buttons, handmask, params, overlap_threshold=0.75):
    width = params["width"]
    blur = params["blur"]

    for name in buttons.keys():
        overlap = cv.bitwise_and(handmask, handmask, mask=buttons[name].mask)

        if np.sum(overlap) / np.sum(buttons[name].mask) > overlap_threshold:
            width, blur = buttons[name].action([width, blur])  
    
    params["width"] = np.clip(width, 200, 1000)
    params["blur"]  = np.clip(blur, 0, 20)

    return params

def detect_faces(frame):
    gray  = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    return face_cascade.detectMultiScale(gray, 1.3, 5)

def blur_faces(frame, faces, params):
    blur = params["blur"]
    k     = 1 + 2*blur
    sigma = 1 +   blur
    
    for (x,y,w,h) in faces:
        blur_box(frame, (x,y), (x+w,y+h), k, sigma)


def main():
    # initiate video stream
    vc = cv.VideoCapture(0)
    time.sleep(2)

    # initiate parameters controled via `buttons`
    params = {
        "width": 640,
        "blur": 0
    }

    # get some ref values
    h_pre,w_pre = 0,0

    while True:
        # read video frame
        _, frame = vc.read()
        aspect_ratio = get_aspect_ratio(frame)
    
        # resize and flip the frame + fix channel order (cv default: BGR)
        frame = adjust_frame(
            frame, (params["width"], int(params["width"] * aspect_ratio)))

        # convert to correct format for tf 
        input_tensor = preprocess_frame(frame)
    
        # get hand detections for current frame
        #(will always make 100 detections sorted by object probability score)
        hand_detections = detect_hands(input_tensor)

        # get hand masks
        handmask = get_hand_masks(
            hand_detections, frame, threshold=0.7,
            show_bbox=int(args["show_hand_bbox"]),
            show_mask=int(args["show_hand_mask"]))

        # update and draw buttons
        add_buttons(frame, buttons)

        # check for button/handmask overlaps and update params if needed      
        params = update_params(buttons, handmask, params)
        
        # get face bbox and update face blur        
        faces = detect_faces(frame)
        blur_faces(frame, faces, params)

        # display final frame
        cv.imshow("press 'q' to quit",  frame[:,:,::-1])
        h_pre,w_pre = frame.shape[:2]
        stopkey = cv.waitKey(1)
        if stopkey == ord("q"):
            break

    vc.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    args = get_args()

    # load buttons and masks
    buttons = mk_buttons()

    # get out-of-the-box face filter from opencv
    face_cascade = cv.CascadeClassifier(FACE_FILTER_PATH)

    # load the inference model for hand detector
    tf.keras.backend.clear_session()
    detect_hands = tf.saved_model.load(DETECTOR_DIR)
    classes = ["hand"]

    main()