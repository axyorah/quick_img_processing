#!/usr/bin/env python3
"""
Created on Tue Aug 20 10:06:39 2019

@author: axeh
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2 as cv
import imutils
from imutils.video import VideoStream
import argparse
import time
#from utils.hand_utils import add_buttons, get_button_masks, blur_box
from utils.pattern_utils import Pattern, Polygon
from utils.perlin_flow import PerlinFlow

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
#face_cascade = cv.CascadeClassifier(PATH_TO_FACE_FILTER)

# load the inference graph for hand detector
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, "rb") as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name="")
    
#%% ---------------------------------------------------------------------------
# load inference model for hand gesture classifier
gesture_model = load_model("dnn/hand_gestures_model_conv_bn_same256.h5")
    
#%% ---------------------------------------------------------------------------
# useful functions
def build_computation_graph():
    # build computation graph
    ops = tf.compat.v1.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops 
                                    for output in op.outputs}
    tensor_dict = {}
        
    for key in ['num_detections', 'detection_boxes', 
                'detection_scores','detection_classes']:
        tensor_name = key + ':0'
        
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.compat.v1.get_default_graph().\
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
    
    image_tensor = tf.compat.v1.get_default_graph().\
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
    bboxes = []
    for box,score in zip(output_dict["detection_boxes"],
                         output_dict["detection_scores"]):
        # ignore detections with low prodiction score
        #(MR: outputs are already sorted by scores!)
        if score < threshold:
            break
        # box is arranged as (ymin,xmin,ymax,xmax) [0,1]
        y1,x1,y2,x2 = (box * np.array([h,w,h,w])).astype(int)
        handmask[y1:y2,x1:x2,:] = 255
        bboxes.append([y1,x1,y2,x2])
        
        # show additional info if requested            
        if show_bbox:
            cv.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
            
    if show_mask:
        cv.imshow("handmask", handmask)
        
    return bboxes
    
def get_custom_pattern(path):
    # get raw vertice coordinates of a custom pattern
    with open(path, 'r') as f:
        lines = f.read().splitlines()    
    x = [int(line.split(' ')[0]) for line in lines]
    y = [int(line.split(' ')[1]) for line in lines]
    
    # normalize vertice coordinates of a cusom pattern
    center = np.array([[360, 376]])
    #xymax = np.array([[700,700]])
    vertices = np.array([x,y]).T 
    vertices -= center # center around origin
    
    rad = max(np.linalg.norm(vertices, axis=1))
    vertices = vertices / rad # fit within unit circle
    
    return vertices

def generate_pattern1(res, dim):
    center = [res[1]//2, res[0]//2] # pattern center
    
    pf = PerlinFlow()
    perlin = pf.get_perlin()
    
    # pents
    n_pent = 10
    pent_rad = 0.3 * dim
    pent_dist = 1.3 * dim
    perlin_idx = np.random.choice(perlin.shape[0])
    pents = [Polygon(5, center, pent_rad, dist_from_center=pent_dist,
                     perlin=perlin, 
                     perlin_idx=perlin_idx)
             for i in range(n_pent)]
    for i,pent in enumerate(pents):       
        pent.initialize_vertices(angles=[2*i * 2*np.pi/5 for i in range(6)])
        pent.initialize_dists()
        pent.rotate_vertices(i * 2*np.pi/n_pent)
        
    # red
    vertices = get_custom_pattern('utils/custom_pattern.txt') * dim 
    reds = [Pattern(vertices, center, dist_from_center=0)]
    
    pattern = reds + pents
    return pattern

def generate_pattern2(res, dim):
    center = [res[1]//2, res[0]//2] # pattern center
    
    pf = PerlinFlow()
    perlin = pf.get_perlin()
    
    # triangles
    n_tri = 3   
    tri_rad = 0.7 * dim; # center of the circle that touches triangle vetices
    perlin_ids = [i * perlin.shape[0]//n_tri for i in range(n_tri)]
    tris = [Polygon(3, center, tri_rad, perlin=perlin, perlin_idx=perlin_ids[i]) 
            for i in range(n_tri)]

    # squares
    n_sq = 6
    sq_rad = 1.1 * dim
    perlin_ids = [np.random.choice(perlin.shape[0])] * n_sq
    squares = [Polygon(4, center, (0.93 if i%2 else 1)*sq_rad, 
                       perlin=perlin, perlin_idx=perlin_ids[i]) 
               for i in range(n_sq)]
    for i in range(0,n_sq,2):
        squares[i  ].rotate_vertices(i//2 * 2*np.pi/(n_sq//2) / 4)
        squares[i+1].rotate_vertices(i//2 * 2*np.pi/(n_sq//2) / 4)

    # hex
    n_hex = 2
    hex_rad = 1.40 * dim
    perlin_ids = [i * perlin.shape[0] // (n_hex//2) 
                  for i in range(n_hex//2) for _ in range(2)]
    hexes   = [Polygon(6, center, (0.93 if i%2 else 1)*hex_rad, 
                       perlin=perlin, perlin_idx=perlin_ids[i]) 
              for i in range(n_hex)]

    # almost circle
    n_circ = 1
    circ_rad = [i * dim for i in [1.15, 1.40, 1.45]]
    circs = [Polygon(18, center, circ_rad[i], perlin=perlin) 
             for i in range(n_circ)]
    
    # mini triangles
    n_mtri = 6
    mtri_rad = 0.2*dim
    mtri_dist = 1.55 * dim
    mtris = [Polygon(3, center, mtri_rad, 
                     dist_from_center=mtri_dist, perlin=perlin)
             for i in range(n_mtri)]
    for i,mtri in enumerate(mtris):
        mtri.rotate_vertices(i * 2*np.pi/n_mtri)  
        
    # pent
    pent_rad = 0.3*dim
    perlin_idx = np.random.choice(perlin.shape[0])
    pents = [Polygon(5, center, pent_rad, 
                     perlin=perlin, 
                     perlin_idx=perlin_idx)]    
    for pent in pents:
        pent.initialize_vertices(angles=[2*i * 2*np.pi/5 for i in range(6)])
        pent.initialize_dists()
        
        
    pattern = tris + squares + hexes + circs + mtris + pents
    return pattern


def add_pattern(frame, pattern, bboxes, fill=False):
    
    h,w = frame.shape[:2]
    dim0 = h//3
    
    # for each hand draw a pattern around it
    for hand in bboxes:
        # get coordinates of hand bbox
        y1,x1,y2,x2 = hand
        
        # for each bbox find its center and 'dimension' (half of main diag)
        center = np.array([[int(0.5*(x1+x2)), int(0.5*(y1+y2))]])
        dim = 0.5 * np.linalg.norm(np.array([x1-x2,y1-y2]))
        
        # for each bbox draw a pattern around it, fig by fig
        for fig in pattern:            
            # update vertices with new center and scale
            fig.update_vertices(center=center, scale=dim/dim0)         
            vertices = fig.vertices.astype(int)
            vertices.reshape((-1,1,2))

            # add updated vertices to frame
            if fill:
                cv.fillPoly(frame, [vertices], (0,0,255))
            else:
                cv.polylines(frame, [vertices], True, (0,0,255), 2)
     
    
    

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

# generate pattern
sz_ref = h//3
pattern1 = generate_pattern1([w, h], sz_ref)
pattern2 = generate_pattern2([w, h], sz_ref)

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
            bboxes = get_hand_masks(tensor_dict, frame, threshold=0.8,
                                    show_bbox=int(args["show_hand_bbox"]),
                                    show_mask=int(args["show_hand_mask"]))
            
            # ----------------------------------------------------------------- 
            # draw the pattern around hands
            add_pattern(frame, pattern1, bboxes, fill=True)
                               
            # -----------------------------------------------------------------
            # display final frame
            cv.imshow("press 'q' to quit",  frame)
            h_pre,w_pre = frame.shape[:2]
            stopkey = cv.waitKey(1)
            if stopkey == ord("q"):
                break
    
cv.destroyAllWindows()
vs.stop()    
