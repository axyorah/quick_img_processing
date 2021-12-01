#!/usr/bin/env python3
"""
Everyone is better off with a tentacle beard!

use:
    $ python3 tentacle_beard.py [-w 0.3]
    
to quit press "q"

template for facial landmarks detection is taken from:
   https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/ 

shape predictor taken from:
    https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat

"""
import cv2 as cv
import numpy as np
import dlib
import argparse
import time

from utils.simple_tentacle import SimpleTentacle, SimpleTentacleBuilder
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
        "-p", "--shapepredictor", 
        default="./dnn/facial/shape_predictor_68_face_landmarks.dat",
        help="path to dlib shape predictor (.dat)\n"+\
            "cat be downloaded from\n"+\
            "https://github.com/AKSHAYUBHAT/TensorFace/blob/master/"+\
            "openface/models/dlib/shape_predictor_68_face_landmarks.dat")

    args = vars(parser.parse_args())
    args["wigl"] = float(args["wigl"])

    return args

class Rect2BBAdapter(list):
    """
    get bounding predicted by dlib
    and convert it (x,y,w,h)
    """
    def __init__(self, rect):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        self.extend([x,y,w,h])

class ShapeConverter:
    """
    coords of 68 facial landmarks as list or numpy array
    """
    NUM_PTS = 68
    def __init__(self, shape, dtype=int):
        self.shape = shape
        self.dtype = dtype

    def to_list(self):
        return [
            [self.shape.part(i).x, self.shape.part(i).y]
            for i in range(self.NUM_PTS)
        ]

    def to_array(self):
        return np.array(self.to_list(), dtype=self.dtype)


class GeoHelper:
    @classmethod
    def midpoint(cls, pt1, pt2, dtype=int): 
        """
        get coords of midpoint of pt1 (x1,y1) and pt2 (x2,y2)
        return the result as (2,) numpy array of ints (pixels!)
        """
        return np.array([
            0.5*(pt1[0]+pt2[0]),
            0.5*(pt1[1]+pt2[1])
        ], dtype=dtype)

    @classmethod
    def dist(cls, pt1, pt2):
        return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)


class BeardTentacleBuilder(SimpleTentacleBuilder):
    def __init__(self, tentacle: 'BeardTentacle'):
        super().__init__(tentacle)
            
    def perlin_idx(self, _perlin_idx):
        self.tentacle._perlin_idx = _perlin_idx
        return self
            
    def color(self, _color):
        self.tentacle._color = _color
        return self
            
    def thickness(self, _thickness):
        self.tentacle._thickness = _thickness
        return self

    def num_joints(self, _num_joints):
        self.tentacle._num_joints = min(
            _num_joints, 
            self.tentacle.MAX_NUM_SEGMENTS+1
        )
        return self

class BeardTentacle(SimpleTentacle):
    MAX_NUM_SEGMENTS = 23
    def __init__(self):
        super().__init__()
        self._perlin_idx = 0
        self._color = (255,0,0)
        self._thickness = [
            8 * 0.95**i for i in range(self.MAX_NUM_SEGMENTS)
        ]
            
    @property
    def set(self):
        return BeardTentacleBuilder(self)

def get_perlin():
    pf = PerlinFlow()\
        .set\
            .ver_grid(5)\
            .hor_grid(6)\
            .points_at_last_octave(6)\
        .build()
    perlin = pf.get_perlin()
    return (perlin - perlin.min()) / (perlin.max() - perlin.min())

def initialize(perlin, max_seg=23, min_seg=10):
    """
    initialize:
        bread tentacles (each composed of `min_seg` to `max_seg` segments)
        scale factors for each tentacle (central one should be thicker than lateral)
        scale factor (thickness) for each tentacle segment    
        indices for each tentacle corresponding to a row in perlin mtx
        color corresponding to each tentacle   
        flip array (all the left-side tentacles should be flipped)
    """
    joint_base  = np.random.randint(min_seg, max_seg, NUM_BEARD_TENTCLS)
    scale_base = [
        13 + 13*np.sin(angle) 
        for angle in np.linspace(0,np.pi,NUM_BEARD_TENTCLS)
    ]
    flip_base = [
        True if i < NUM_BEARD_TENTCLS//2 else False
        for i in range(NUM_BEARD_TENTCLS)
    ]
    perlin_base  = np.random.choice(range(perlin.shape[0]), NUM_BEARD_TENTCLS)
    color_base  = [
        (np.random.randint(100,200), np.random.randint(100,230), 0)
        for _ in range(NUM_BEARD_TENTCLS)
    ]
    thickness_base = [8 * 0.95**i for i in range(max_seg)]

    return [
        BeardTentacle()\
            .set\
                .num_joints(joint_base[i])\
                .scale(scale_base[i])\
                .flip(flip_base[i])\
                .perlin_idx(perlin_base[i])\
                .color(color_base[i])\
                .thickness(thickness_base[:joint_base[i]])
            .build()
        for i in range(NUM_BEARD_TENTCLS)
    ]


def get_face_scaling_factor(landmarks):
    """
    get relative face width (face width to frame width) wrt reference
    """
    facedim = max(
        GeoHelper.dist(landmarks[3], landmarks[13]),
        GeoHelper.dist(landmarks[8], landmarks[27])
    ) / FRAME_WIDTH
    face_scale = facedim / FACEDIM_REF
    return face_scale

def draw_tentacle(
        frame, frame_idx, 
        tentacle, landmark, 
        center, scale
    ):    
    # get direction of arm_angle:            
    #(direction from highest nose point and anchor of a beard tentacle) 
    x = landmark[0] - center[0]
    y = landmark[1] - center[1]
            
    # sample perlin mtx for smooth "randomness"
    perlin_random = perlin[
        tentacle._perlin_idx, 
        frame_idx % perlin.shape[1]
    ]

    scale_ref = tentacle._scale # cache scale!
    tentacle\
        .set\
            .scale(int(np.round(scale * tentacle._scale)))\
            .root(landmark)\
            .arm_angle((x,y))\
            .max_angle_between_segments(args["wigl"]*np.pi * perlin_random)\
            .angle_freq(1 * perlin_random)\
            .angle_phase_shift(2*np.pi * perlin_random)\
        .build()

    coords = tentacle.solve().astype(int)
    
    for i in range(coords.shape[1]-1):
        cv.line(
            frame, 
            tuple(coords[:,i]), 
            tuple(coords[:,i+1]), 
            tentacle._color, 
            int(np.round(scale * tentacle._thickness[i]))
        )

    # reset scale!!!
    tentacle.set.scale(scale_ref)

def draw_mustachio(
        frame, frame_idx, 
        tentacle, anchor, 
        center, scale
    ):

    y = anchor[1] - center[1]
    x = anchor[0] - center[0] + 1e-16

    perlin_random = perlin[
        tentacle._perlin_idx, 
        frame_idx % perlin.shape[1]
    ]

    scale_ref = tentacle._scale # cache init scale!
    tentacle\
        .set\
            .scale(int(scale * 9))\
            .root(anchor)\
            .arm_angle((x,y))\
            .max_angle_between_segments(np.pi/5 * perlin_random)\
            .angle_freq(1 * perlin_random)\
            .angle_phase_shift(np.pi * perlin_random)\
        .build()

    coords = tentacle.solve().astype(int)

    for i in range(coords.shape[1]-1):
        cv.line(
            frame, 
            tuple(coords[:,i]), 
            tuple(coords[:,i+1]), 
            (100,100,0), 
            int(np.round(scale * 5))
        )
    # reset scale!!!
    tentacle.set.scale(scale_ref)
        

def draw_brows(frame, landmarks, lndmrk_idx_start, lndmrk_idx_end, scale):
    for i in range(lndmrk_idx_start, lndmrk_idx_end+1):
        cv.line(
            frame, 
            tuple(landmarks[i]), 
            tuple(landmarks[i+1]), 
            (100,100,0), 
            int(np.round(scale * 7))
        )

def main():
    # start video stream
    vc = cv.VideoCapture(0)
    time.sleep(2)

    # initialize tentacles
    tentacles = initialize(perlin)

    frame_idx = 0 # will be used to sample perlin mtx
    while True:
        _, frame = vc.read()
        h,w,_ = frame.shape
        frame = cv.resize(frame, (FRAME_WIDTH, FRAME_WIDTH * h // w))
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
        # get bounding boxes for face detections
        #(or use the ones from the previous step)
        faces = detector(gray, 1)
    
        # go through each detected face
        for face in faces:
            # get 68 facial landmarks for current face
            landmarks_raw = predictor(gray, face)
            landmarks     = ShapeConverter(landmarks_raw).to_array()
        
            # get central beard landmark (highest nose point)
            #(all beard tentacles are poining away from it)
            beard_center = landmarks[27]

            # get central mustache landmark
            #(mustache 'branches' are pointing away from it)
            must_center = GeoHelper.midpoint(landmarks[33], landmarks[51])
        
            # estimate face dimension relative to the frame
            face_scale = get_face_scaling_factor(landmarks)
        
            # use landmarks 3-16 to draw tentacle beard
            for i in range(NUM_BEARD_TENTCLS):
                draw_tentacle(
                    frame, frame_idx, 
                    tentacles[i], landmarks[i + 2], 
                    beard_center, face_scale
                )
            
            # use landmarks 33-35, 51-53 to draw mustache
            # left mustachio
            left_anchor = GeoHelper.midpoint(landmarks[32], landmarks[50])
            draw_mustachio(
                frame, frame_idx, 
                tentacles[0], left_anchor, 
                must_center, face_scale
            )
            
            # right mustachio
            right_anchor = GeoHelper.midpoint(landmarks[34], landmarks[52])
            draw_mustachio(
                frame, frame_idx, 
                tentacles[-1], right_anchor,  
                must_center, face_scale
            )
            
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
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shapepredictor"])

    # get perlin mtx for smooth randomness
    perlin = get_perlin()

    main()