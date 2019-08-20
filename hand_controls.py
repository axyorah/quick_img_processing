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
PATH_TO_FROZEN_GRAPH = "./dnn/frozen_inference_graph_for_hand_detection.pb"
PATH_TO_FACE_FILTER  = "./dnn/haarcascade_frontalface_default.xml"

#%% ---------------------------------------------------------------------------
# get out-of-the-box face filter from opencv
face_cascade = cv.CascadeClassifier(PATH_TO_FACE_FILTER)

# load the inference graph for hand detector
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, "rb") as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name="")
    
#%% ---------------------------------------------------------------------------
# useful functions
def build_computation_graph():
    # build computation graph
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops 
                                    for output in op.outputs}
    tensor_dict = {}
        
    for key in ['num_detections', 'detection_boxes', 
                'detection_scores','detection_classes']:
        tensor_name = key + ':0'
        
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().\
                                  get_tensor_by_name(tensor_name)
        
    # get detection boxes
    if 'detection_boxes' in tensor_dict:
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], 
                                     tf.int32)
        detection_boxes = tf.slice(detection_boxes, 
                                   [0, 0], 
                                   [real_num_detection, -1])
    
    image_tensor = tf.get_default_graph().\
                      get_tensor_by_name('image_tensor:0')
    return image_tensor, tensor_dict

def get_hand_masks(tensor_dict, frame, threshold=0.5,
                   show_bbox=0, show_mask=0):
    # convert BGR (opencv default) to RGB and add batch dimension
    frame4d = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame4d = np.expand_dims(frame4d, axis=0)
    
    # run inference to get hand detections
    output_dict = sess.run(tensor_dict, 
                           feed_dict={image_tensor: frame4d})
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    
    # iterate over the detections to get hand masks
    handmask = np.zeros(frame.shape)
    for box,score in zip(output_dict["detection_boxes"],
                         output_dict["detection_scores"]):
        # ignore detections with low prodiction score
        #(MR: outputs are already sorted by scores!)
        if score < threshold:
            break
        # box is arranged as (ymin,xmin,ymax,xmax) [0,1]
        y1,x1,y2,x2 = (box * np.array([h,w,h,w])).astype(int)
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
width = 500
blur  = 0

# sample video stream to get some params
sample = vs.read()
sample = imutils.resize(sample, width=width)
h,w = sample.shape[:2]
aspect_ratio = h/w

# get some ref values
h_pre,w_pre = 0,0

with detection_graph.as_default():
    with tf.compat.v1.Session() as sess:
        
        # ---------------------------------------------------------------------
        # build computation graph
        image_tensor, tensor_dict = build_computation_graph()
    
        # ---------------------------------------------------------------------
        # capture/process frames from the video stream
        while True:
            frame = vs.read()
            frame = cv.resize(frame, (width, int(width * aspect_ratio)))
            frame = cv.flip(frame, 1)       
            gray  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            h,w   = frame.shape[:2]
            
            # ----------------------------------------------------------------- 
            # run inference to get hand masks   
            handmask = get_hand_masks(tensor_dict, frame, threshold=0.5,
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
            cv.imshow("press 'q' to quit",  frame)
            h_pre,w_pre = frame.shape[:2]
            stopkey = cv.waitKey(1)
            if stopkey == ord("q"):
                break
    
cv.destroyAllWindows()
vs.stop()    