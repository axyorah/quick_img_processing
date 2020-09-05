#!/usr/bin/env python3
"""
Created on Thu Aug  1 15:37:59 2019

Everyone is better off with a tentacle beard!

use:
    $ python3 tentacle_beard.py [-w 0.3]
    
to quit press "q"

template for facial landmarks detection is taken from:
   https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/ 

shape predictor taken from:
    https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat
    
to install dlib on windows use wheel:
    https://pypi.python.org/packages/da/06/bd3e241c4eb0a662914b3b4875fc52dd176a9db0d4a2c915ac2ad8800e9e/dlib-19.7.0-cp36-cp36m-win_amd64.whl#md5=b7330a5b2d46420343fbed5df69e6a3f

@author: axeh
"""

import cv2 as cv
import numpy as np
import dlib
import imutils
from imutils.video import VideoStream
import argparse
import time

from utils.simple_tentacle import SimpleTentacle
from utils.perlin_flow import PerlinFlow

#%% ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-w", "--wigl", type=float,
                    default=0.25,
                    help="degrgee of tentacle 'wiggliness':\n"+\
                         "should be a float from 0+ to 0.5")
parser.add_argument(
        "-p", "--shapepredictor", 
        default="./dnn/shape_predictor_68_face_landmarks.dat",
        help="path to dlib shape predictor (.dat)\n"+\
             "cat be downloaded from\n"+\
             "https://github.com/AKSHAYUBHAT/TensorFace/blob/master/"+\
             "openface/models/dlib/shape_predictor_68_face_landmarks.dat")
args = vars(parser.parse_args())
args["wigl"] = float(args["wigl"])

#%% ---------------------------------------------------------------------
# useful stuff for face/facial landmarks detection
def rect2bb(rect):
    """
    get bounding predicted by dlib
    and convert it (x,y,w,h)
    """
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x,y,w,h)

def shape2np(shape, dtype=int):
    """
    coords of 68 facial landmarks -> numpy array
    """
    coords = np.zeros((68,2), dtype=dtype)
    
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
                
    return coords

def midpoint(pt1, pt2):
    """
    get coords of midpoint of pt1 (x1,y1) and pt2 (x2,y2)
    return the result as (2,) numpy array of ints (pixels!)
    """
    return np.array([int(0.5*(pt1[0]+pt2[0])), 
                     int(0.5*(pt1[1]+pt2[1]))])
    
def dist(pt1, pt2):
    return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
    

# initiate shape detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shapepredictor"])


#%% ---------------------------------------------------------------------
# generate perlin noise for smooth *randomness*
pf = PerlinFlow(ver_grid=5, hor_grid=6, points_at_last_octave=6)
perlin = pf.get_perlin()
perlin = (perlin - perlin.min()) / (perlin.max() - perlin.min())


facedim_ref = 100/500 # ref val for characteristic face width (jaw width)
beard_tentacles = 13  # num of beard tentacles // don't change it...
max_seg = 23          # max num of tentacle segments 
min_seg = 10          # min num of tentacle segments

# base values for some characteristic features
joint_base     = np.random.randint(min_seg,max_seg, beard_tentacles)
tentacle_base  = [SimpleTentacle(joint_base[i])
                  for i in range(beard_tentacles)]
scale_base     = [13 + 13*np.sin(angle) 
                  for angle in np.linspace(0,np.pi,beard_tentacles)]
flip_base      = [True if i < beard_tentacles//2 else False
                  for i in range(beard_tentacles)]
perlin_base    = np.random.choice(range(perlin.shape[0]), beard_tentacles)

color_base     = [(np.random.randint(100,200), np.random.randint(100,230), 0)
                  for _ in range(beard_tentacles)]
thickness_base = [8 * 0.95**i for i in range(max_seg)]

# start video stream
vc = cv.VideoCapture(0)
time.sleep(2)

idx = 0 # will be used to sample perlin mtx
prev_faces = None # will be used to smoothen frame transitions
frame_width = 500
while True:
    _, frame = vc.read()
    frame = imutils.resize(frame, width=frame_width)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # get bounding boxes for face detections
    #(or use the ones from the previous step)
    faces = detector(gray, 1)
    #if not faces and prev_faces is not None:
    #   faces = prev_faces
    
    # go through each detected face        
    for iface,face in enumerate(faces):   
        # get 68 facial landmarks for current face
        landmarks_raw = predictor(gray, face)
        landmarks     = shape2np(landmarks_raw)        
        #for landmark in landmarks:
        #    cv.circle(frame, tuple(landmark), 1, (0,0,255), -1) 
        
        # get central beard landmark (highest nose point)
        #(all beard tentacles are poining away from it)
        center = landmarks[27]
        
        # estimate face dimension relative to the frame
        facedim = max(dist(landmarks[3], landmarks[13]),
                      dist(landmarks[8], landmarks[27])) / frame_width
        face_scale = facedim / facedim_ref
        
        # use landmarks 3-16 to draw tentacle beard            
        for it,lndmrk in zip(range(beard_tentacles), range(2,16,1)):
            # get direction of arm_angle:            
            #(direction from highest nose point and root of a berad tentacle) 
            x = landmarks[lndmrk][0] - center[0]
            y = landmarks[lndmrk][1] - center[1]
            
            # sample perlin mtx for smooth "randomness"
            perlin_random = perlin[perlin_base[it],
                                   idx % perlin.shape[1]]
            
            # wiggle the tentacle smoothly and "randomly"
            t = tentacle_base[it].get_wiggly_tentacle(
                    int(np.round(face_scale * scale_base[it])),
                    landmarks[lndmrk],
                    (x,y),
                    max_angle_between_segments=args["wigl"]*np.pi * perlin_random,
                    angle_freq=1 * perlin_random,
                    angle_phase_shift=2*np.pi * perlin_random,
                    flip=flip_base[it]
                    ).astype(int)


            for i in range(t.shape[1]-1):
                cv.line(frame, tuple(t[:,i]), tuple(t[:,i+1]), 
                        color_base[it], 
                        int(np.round(face_scale * thickness_base[i])))
               
        # use landmarks 33-35, 51-53 to draw mustache
        # left mustachio
        y = landmarks[51][1] - landmarks[52][1]
        x = landmarks[51][0] - landmarks[52][0] + 1e-16
        perlin_random = perlin[42 % perlin.shape[0],
                               idx % perlin.shape[1]]
        ml = tentacle_base[0].get_wiggly_tentacle(
                    int(np.round(face_scale * 9)),
                    midpoint(landmarks[32], landmarks[50]),
                    (x,y),
                    max_angle_between_segments=np.pi/5 * perlin_random,
                    angle_freq=1 * perlin_random,
                    angle_phase_shift=np.pi * perlin_random,
                    flip=True
                    ).astype(int)

        for i in range(ml.shape[1]-1):
            cv.line(frame, tuple(ml[:,i]), tuple(ml[:,i+1]), (100,100,0), 
                    int(np.round(face_scale * 5)))
            
        # right mustachio
        y = landmarks[53][1] - landmarks[52][1]
        x = landmarks[53][0] - landmarks[52][0] + 1e-16
        perlin_random = perlin[47 % perlin.shape[0],
                               idx % perlin.shape[1]]
        mr = tentacle_base[-1].get_wiggly_tentacle(
                    int(np.round(face_scale * 9)),
                    midpoint(landmarks[34], landmarks[52]),
                    (x,y),
                    max_angle_between_segments=0.2*np.pi * perlin_random,
                    angle_freq=1 * perlin_random,
                    angle_phase_shift=np.pi * perlin_random,
                    flip=False
                    ).astype(int)

        for i in range(mr.shape[1]-1):
            cv.line(frame, tuple(mr[:,i]), tuple(mr[:,i+1]), (100,100,0), 
                    int(np.round(face_scale * 5)))
            
        # draw brows
        for i in range(17,21):
            cv.line(frame, tuple(landmarks[i]), tuple(landmarks[i+1]), 
                   (100,100,0), int(np.round(face_scale * 7)))
        for i in range(22,26):
            cv.line(frame, tuple(landmarks[i]), tuple(landmarks[i+1]), 
                   (100,100,0), int(np.round(face_scale * 7)))
        
    if faces:
        prev_faces = faces
    
    idx += 1
    cv.imshow("press 'q' to quit", frame)
    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break
vc.release()
cv.destroyAllWindows()

