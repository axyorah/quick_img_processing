#!/usr/bin/env python3
"""
Everyone is better off with a tentacle beard!

use:
    $ python3 tentacle_beard.py [-w 0.3]
    
to quit press "q"

high cpu use, high false positives rate...
"""
import cv2 as cv
import numpy as np
import argparse
import time

from utils.simple_tentacle import SimpleTentacle
from utils.perlin_flow import PerlinFlow

NUM_BEARD_TENTCLS = 13 # hardcoded as it corresponds to 13/15 facial anchor points
FACEDIM_REF = 1/5      # default face width relative to frame width (used to scale the breard)
FRAME_WIDTH = 640

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w", "--wigl", type=float, default=0.25,
        help="degrgee of tentacle 'wiggliness':\n"+\
            "should be a float from 0+ to 0.5")
    parser.add_argument(
        "-p", "--face_prototxt", type=str, 
        default="dnn/facial/face_deploy.prototext",
        help="path to face detector .prototxt file (architecture)"
    )
    parser.add_argument(
        "-m", "--face_model", type=str,
        default="dnn/facial/face_res10_300x300_ssd_iter_140000.caffemodel",
        help="path to face detector caffe model (weights)"
    )
    parser.add_argument(
        "-l", "--landmarks_model",
        default="dnn/facial/lbfmodel.yaml",
        help="path for facial landmark detector (yaml file)"
    )
    args = vars(parser.parse_args())
    args["wigl"] = float(args["wigl"])

    return args

def midpoint(pt1, pt2):
    """
    get coords of midpoint of pt1 (x1,y1) and pt2 (x2,y2)
    return the result as (2,) numpy array of ints (pixels!)
    """
    return np.array([int(0.5*(pt1[0]+pt2[0])), 
                     int(0.5*(pt1[1]+pt2[1]))])
    
def dist(pt1, pt2):
    return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
    
def get_perlin():
    pf = PerlinFlow(ver_grid=5, hor_grid=6, points_at_last_octave=6)
    perlin = pf.get_perlin()
    perlin = (perlin - perlin.min()) / (perlin.max() - perlin.min())
    
    h,w = perlin.shape
    perlin = cv.resize(perlin, (w*3, h*3))
    return perlin

def init(perlin, max_seg=23, min_seg=10):
    """
    initialize:
        bread tentacles (each composed of `min_seg` to `max_seg` segments)
        scale factors for each tentacle (central one should be thicker than lateral)
        scale factor (thickness) for each tentacle segment    
        indices for each tentacle corresponding to a row in perlin mtx
        color corresponding to each tentacle   
        flip array (all the left-side tentacles should be flipped)
    """
    # base values for some characteristic features
    joint_base     = np.random.randint(min_seg,max_seg, NUM_BEARD_TENTCLS)
    tentacle_base  = [SimpleTentacle(joint_base[i])
                      for i in range(NUM_BEARD_TENTCLS)]
    scale_base     = [13 + 13*np.sin(angle) 
                      for angle in np.linspace(0,np.pi,NUM_BEARD_TENTCLS)]
    flip_base      = [True if i < NUM_BEARD_TENTCLS//2 else False
                      for i in range(NUM_BEARD_TENTCLS)]
    perlin_base    = np.random.choice(range(perlin.shape[0]), NUM_BEARD_TENTCLS)
    color_base     = [(np.random.randint(100,200), np.random.randint(100,230), 0)
                      for _ in range(NUM_BEARD_TENTCLS)]
    thickness_base = [8 * 0.95**i for i in range(max_seg)]

    return {
        "tentacle_base": tentacle_base,
        "scale_base": scale_base,
        "flip_base": flip_base,
        "perlin_base": perlin_base,
        "color_base": color_base,
        "thickness_base": thickness_base
    }

def get_face_scaling_factor(landmarks):
    """
    get relative face width (face width to frame width) wrt reference
    """
    facedim = max(
        dist(landmarks[3], landmarks[13]),
        dist(landmarks[8], landmarks[27])) / FRAME_WIDTH
    face_scale = facedim / FACEDIM_REF
    return face_scale

def draw_tentacle_by_idx(
    frame, frame_idx, landmarks, lndmrk_idx, center, scale):

    tentacle_base = params["tentacle_base"]
    color_base = params["color_base"]    
    thickness_base = params["thickness_base"]
    scale_base = params["scale_base"]
    flip_base = params["flip_base"]
    perlin_base = params["perlin_base"]

    # 13 facial landmarks (indices 3 to 15, base-1 indexing) 
    # are used as the anchor points to draw breard tentacle;
    # tentacle indices are simple landmark_idx - 2
    idx = lndmrk_idx - 2 
    
    # get direction of arm_angle:            
    #(direction from highest nose point and anchor of a beard tentacle) 
    x = landmarks[lndmrk_idx][0] - center[0]
    y = landmarks[lndmrk_idx][1] - center[1]
            
    # sample perlin mtx for smooth "randomness"
    perlin_random = perlin[perlin_base[idx], frame_idx % perlin.shape[1]]

    t = tentacle_base[idx].get_wiggly_tentacle(
        int(np.round(scale * scale_base[idx])),
        landmarks[lndmrk_idx],
        (x,y),
        max_angle_between_segments=args["wigl"]*np.pi * perlin_random,
        angle_freq=1 * perlin_random,
        angle_phase_shift=2*np.pi * perlin_random,
        flip=flip_base[idx]
        ).astype(int)


    for i in range(t.shape[1]-1):
        line_thickness = int(np.round(scale * thickness_base[i]))
        line_thickness = max(1, line_thickness)
        line_thickness = min(16, line_thickness)
        cv.line(frame, tuple(t[:,i]), tuple(t[:,i+1]), 
                color_base[idx], line_thickness)

def draw_mustachio(
    frame, frame_idx, anchor, landmarks, lndmrk_idx, must_idx, 
    center, scale, flip):
    tentacle_base = params["tentacle_base"]

    y = landmarks[lndmrk_idx][1] - center[1]
    x = landmarks[lndmrk_idx][0] - center[0] + 1e-16
    perlin_random = perlin[must_idx % perlin.shape[0],
                           frame_idx % perlin.shape[1]]

    m = tentacle_base[0].get_wiggly_tentacle(
        int(scale * 9), anchor, (x,y),
        max_angle_between_segments=np.pi/5 * perlin_random,
        angle_freq=1 * perlin_random,
        angle_phase_shift=np.pi * perlin_random,
        flip=flip
        ).astype(int)

    for i in range(m.shape[1]-1):
        cv.line(frame, tuple(m[:,i]), tuple(m[:,i+1]), (100,100,0), 
                int(np.round(scale * 5)))

def draw_brows(frame, landmarks, lndmrk_idx_start, lndmrk_idx_end, scale):
    for i in range(lndmrk_idx_start,lndmrk_idx_end+1):
        cv.line(frame, tuple(landmarks[i]), tuple(landmarks[i+1]), 
               (100,100,0), int(np.round(scale * 7)))

def main():
    # start video stream
    vc = cv.VideoCapture(0)
    time.sleep(2)

    frame_idx = 0 # will be used to sample perlin mtx
    while True:
        _, frame = vc.read()
        h,w,_ = frame.shape
        frame = cv.resize(frame, (FRAME_WIDTH, FRAME_WIDTH * h // w))
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blob = cv.dnn.blobFromImage(
            cv.resize(frame, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0)
        )

        # get bounding boxes for face detections
        detector.setInput(blob)
        detections = detector.forward()

        faces = []
        for i in range(detections.shape[2]):
            p, x1, y1, x2, y2 = detections[0,0,i,2:7]
            if p < 0.5:
                continue
            x1,x2 = map(lambda x: int(x * w), [x1,x2])
            y1,y2 = map(lambda y: int(y * h), [y1,y2])
            faces.append([x1, y1, x2-x1, y2-y1])

        if not len(faces):
            landmarks_all = []
        else:
            _,landmarks_all = predictor.fit(gray, np.array(faces, dtype=np.int32))

        # go through each detected face
        for landmarks in landmarks_all:
            # get 68 facial landmarks for current face
            landmarks = (landmarks[0]).astype(int)
        
            # get central beard landmark (highest nose point)
            #(all beard tentacles are poining away from it)
            beard_center = landmarks[27]

            # get central mustache landmark
            #(mustache 'branches' are pointing away from it)
            must_center = landmarks[52]
        
            # estimate face dimension relative to the frame
            face_scale = get_face_scaling_factor(landmarks)
        
            # use landmarks 3-16 to draw tentacle beard            
            for lndmrk_idx in range(2,2+NUM_BEARD_TENTCLS,1):             
                draw_tentacle_by_idx(
                    frame, frame_idx, landmarks, lndmrk_idx, 
                    beard_center, face_scale)
                
               
            # use landmarks 33-35, 51-53 to draw mustache
            # left mustachio
            left_anchor = midpoint(landmarks[32], landmarks[50])
            draw_mustachio(
                frame, frame_idx, left_anchor, landmarks, 51, 42, 
                must_center, face_scale, True)
            
            # right mustachio
            right_anchor = midpoint(landmarks[34], landmarks[52])
            draw_mustachio(
                frame, frame_idx, right_anchor, landmarks, 53, 47, 
                must_center, face_scale, False)
            
            # draw brows
            draw_brows(frame, landmarks, 17, 20, face_scale)
            draw_brows(frame, landmarks, 22, 25, face_scale)
        
        frame_idx += 1
        cv.imshow("press 'q' to quit", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    vc.release()
    cv.destroyAllWindows()


if __name__ == "__main__":    
    args = get_args()
    
    # initiate face detector and facial landmark predictor
    detector = cv.dnn.readNetFromCaffe(args["face_prototxt"], args["face_model"])

    predictor = cv.face.createFacemarkLBF()
    predictor.loadModel(args["landmarks_model"])

    # get perlin mtx for smooth randomness
    perlin = get_perlin()

    # initialize tentacles
    params = init(perlin)

    main()