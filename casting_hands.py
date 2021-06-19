#!/usr/bin/env python3
import torch
import cv2 as cv
import time

from utils.pattern_utils import \
    FistPatternEffect, HandPatternEffect, \
    JutsuPatternEffect, LightningPatternEffect
from utils.yolo_utils_by_ultralytics.yolo import Model

DETECTOR_DICT = "./dnn/yolov5_gesture_state_dict.pt"
DETECTOR_YAML = "./dnn/yolov5s.yaml"
IMGSZ = 448
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
HALF = False

def load_model(cfg_path, state_dict_path, device=torch.device('cpu'), half=False):
    # restore model from state_dict checkpoint
    detector = Model(cfg=cfg_path, nc=5)
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

def adjust_frame(frame, tar_sz):
    frame = cv.resize(frame, tar_sz)
    frame = cv.flip(frame, 1)
    return frame

def preprocess_frame(frame, sz, device='cpu', half=False):
    tensor = cv.resize(frame, sz)    
    tensor = tensor.transpose(2, 0, 1)
    tensor = torch.from_numpy(tensor)
    tensor = torch.unsqueeze(tensor, 0)
    tensor = tensor.half() if half else tensor.float()
    tensor = tensor / 255.0
    tensor = tensor.to(device)
    return tensor

def draw_effects(frame, detections, classes):
    h,w = frame.shape[:2]

    jutsu_detected, lightning_detected = False, False
    jutsu_pt1, jutsu_pt2 = (None,None), (None,None) #TODO: make explosion appear from the center of the box
    lightning_pt1, lightning_pt2 = (None,None), (None,None)
    
    for x1,y1,x2,y2,prob,clss in detections:
        # get box coordinates
        x1,x2 = map(lambda x: int(x * w / IMGSZ), [x1, x2])        
        y1,y2 = map(lambda y: int(y * h / IMGSZ), [y1, y2])
        clss = int(clss)

        # Recall: class indices start from 1 (0 is reserved for background)
        if classes[clss] == "hand":
            handpattern.draw_pattern(frame, (x1,y1), (x2,y2))
        elif classes[clss] == "fist":
            fistpattern.draw_pattern(frame, (x1,y1), (x2,y2))            
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
    jutsupattern.draw_pattern(
        frame, jutsu_detected, jutsu_pt1, jutsu_pt2)
    lightningpattern.draw_pattern(
        frame, lightning_detected, lightning_pt1, lightning_pt2)


def main():
    # adjust float precision: if no cuda - alawys use float32 instead of float16/float32
    half = False if DEVICE.type == 'cpu' else HALF

    # load model
    detector = load_model(DETECTOR_YAML, DETECTOR_DICT, device=DEVICE, half=half)
    classes = ['fist', 'hand', 'horns', 'teleportation_jutsu', 'tori_sign']

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
        input_tensor = preprocess_frame(frame, (IMGSZ, IMGSZ), device=DEVICE, half=half)
    
        # get detections for current frame
        detections = detector(input_tensor)[0]

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
    fistpattern = FistPatternEffect()
    handpattern = HandPatternEffect()
    jutsupattern = JutsuPatternEffect()   
    lightningpattern = LightningPatternEffect()  

    main()