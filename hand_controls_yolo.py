#!/usr/bin/env python3

import numpy as np
import torch
import cv2 as cv
import argparse
import time
from utils.hand_utils import mk_buttons, add_buttons, blur_box
from utils.yolo_utils_by_ultralytics.yolo import Model

FACE_FILTER_PROTO = "./dnn/facial/face_deploy.prototext"
FACE_FILTER_RCNN = "./dnn/facial/face_res10_300x300_ssd_iter_140000.caffemodel"
HAND_DETECTOR_YAML = "./dnn/yolov5s.yaml"
HAND_DETECTOR_DICT = "./dnn/yolov5_palm_state_dict.pt"

IMGSZ = 448
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
HALF = False

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

def load_yolo_model(cfg_path, state_dict_path, num_classes, device=torch.device('cpu'), half=False):
    # restore model from state_dict checkpoint
    detector = Model(cfg=cfg_path, nc=num_classes)
    ckpt = torch.load(state_dict_path)
    detector.load_state_dict(ckpt['model_state_dict'])

    # add non-max suppression
    detector.nms()

    # adjust precision
    if half:
        detector.half()
    else:
        detector.float()

    # use cuda is available
    detector.to(device)

    # switch to inference mode
    detector.eval()
    return detector

def get_hand_masks(detections, frame, threshold=0.5,
                   show_bbox=0, show_mask=0):
    h,w = frame.shape[:2]
    handmask = np.zeros(frame.shape)
    for x1,y1,x2,y2,prob,clss in detections:
        if prob < 0.5:
            continue
        # get box coordinates
        x1,x2 = map(lambda x: int(x * w / IMGSZ), [x1, x2])        
        y1,y2 = map(lambda y: int(y * h / IMGSZ), [y1, y2])
        clss = int(clss)

        # get box coordinates
        #y1,x1,y2,x2 = (box*np.array([h,w,h,w])).astype(int)
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

def preprocess_frame(frame, sz, device=None, half=False):
    if device is None:
        device = torch.device('cpu')
    tensor = cv.resize(frame, sz)    
    tensor = tensor.transpose(2, 0, 1)
    tensor = torch.from_numpy(tensor)
    tensor = torch.unsqueeze(tensor, 0)
    tensor = tensor.half() if half else tensor.float()
    tensor = tensor / 255.0
    tensor = tensor.to(device)
    return tensor

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
    blob = cv.dnn.blobFromImage(
        cv.resize(frame, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0)
    )
    face_detector.setInput(blob)
    face_detections = face_detector.forward()
    faces = []
    for i in range(face_detections.shape[2]):
        p,x1,y1,x2,y2 = face_detections[0,0,i,2:7]
        if p < 0.5:
            continue
        x1,x2 = map(lambda x: int(x * frame.shape[1]), [x1,x2])
        y1,y2 = map(lambda y: int(y * frame.shape[0]), [y1,y2])
        faces.append([x1, y1, x2-x1, y2-y1])
    return faces

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
        #input_tensor = preprocess_frame(frame)
        input_tensor = preprocess_frame(frame, (IMGSZ, IMGSZ), device=DEVICE, half=half)
    
        # get hand detections for current frame
        #(will always make 100 detections sorted by object probability score)
        #hand_detections = detect_hands(input_tensor)
        hand_detections = hand_detector(input_tensor)[0]

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

    # adjust float precision: if no cuda - use float32
    half = False if DEVICE.type == 'cpu' else HALF

    # load buttons and masks
    buttons = mk_buttons()

    # get out-of-the-box face filter from opencv
    face_detector = cv.dnn.readNetFromCaffe(
        FACE_FILTER_PROTO,
        FACE_FILTER_RCNN
    )

    # load the inference model for hand detector
    hand_detector = load_yolo_model(
        HAND_DETECTOR_YAML,
        HAND_DETECTOR_DICT,
        num_classes=1, device=DEVICE, half=half
    )
    classes = ["hand"]

    main()