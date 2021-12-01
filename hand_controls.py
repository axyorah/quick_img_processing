#!/usr/bin/env python3

import numpy as np
import torch
import cv2 as cv
import argparse
import time
#from utils.hand_utils import mk_buttons, add_buttons, blur_box
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

def get_hand_masks(
    detections, 
    frame, 
    threshold=0.5,
    show_bbox=0, 
    show_mask=0
):
    h,w = frame.shape[:2]
    handmask = np.zeros(frame.shape)
    for x1, y1, x2, y2, prob, clss in detections:
        if prob < threshold:
            continue
        # get box coordinates
        x1, x2 = map(lambda x: int(x * w / IMGSZ), [x1, x2])        
        y1, y2 = map(lambda y: int(y * h / IMGSZ), [y1, y2])
        clss = int(clss)

        # get box coordinates
        handmask[y1:y2, x1:x2, :] = 255

        # show additional info if requested
        if show_bbox:
            cv.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)

    if show_mask:
        cv.imshow("handmask", handmask)

    return handmask

def blur_box(frame, pt1, pt2, k, sigma, weight=0.7):
    xmin,ymin = pt1
    xmax,ymax = pt2

    blur     = frame.copy()
    blur     = cv.GaussianBlur(blur, (k,k), sigma)

    frame[ymin:ymax,xmin:xmax,:] = blur[ymin:ymax,xmin:xmax,:]

    return frame

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

def blur_faces(frame, blur):
    faces = detect_faces(frame)    
    k     = 1 + 2*blur
    sigma = 1 +   blur
    
    for (x,y,w,h) in faces:
        blur_box(frame, (x,y), (x+w,y+h), k, sigma)
        
    return frame
        
def resize_frame(frame, width):
    width = int(width)
    height = int(FrameManager.get_aspect_ratio(frame) * width)
    return cv.resize(frame, (width, height))

    
class Event(list):
    def __call__(self, *args, **kwargs):
        for item in self:
            item(*args, **kwargs)

class Observable:
    def __init__(self):
        self.change = Event()

class Param:
    def __init__(
        self, 
        name, 
        val, 
        min_val, 
        max_val, 
        effect_fun
    ):
        self.name = name
        self.val = val
        self.min_val = min_val
        self.max_val = max_val
        self.effect_fun = effect_fun

    def update(self, frame, param, amount_change):
        if param == self:
            self.val += amount_change
            self.val = np.clip(self.val, self.min_val, self.max_val)
            self.effect_fun(frame, self.val)

class Button(Observable):
    def __init__(self, name, path, param, amount_change, persist=False):
        super().__init__()
        self.name = name
        self.path = path
        self.param = param
        self.amount_change = amount_change
        self.persist = persist
        self.btn_ref = None
        self.mask_ref = None
        self.btn = None
        self.mask = None
        self.load()
        self.change.append(self.param.update)

    def load(self):
        comb = cv.imread(self.path, cv.IMREAD_UNCHANGED)
        comb = cv.cvtColor(comb, cv.COLOR_BGRA2RGBA)
        self.btn_ref = comb[:,:,:3]
        self.mask_ref = comb[:,:,3]
        self.btn = self.btn_ref.copy()
        self.mask = self.mask_ref.copy()

    def add(self, frame):
        # resize if needed
        h,w,c = frame.shape
        if self.mask != (h,w):
            self.btn = cv.resize(self.btn_ref, (w,h))
            self.mask = cv.resize(self.mask_ref, (w,h))

        fg = cv.bitwise_and(self.btn, self.btn, mask=self.mask)
        bg = cv.bitwise_and(frame, frame, mask=cv.bitwise_not(self.mask))
        cv.add(bg, fg, dst=frame)

    def resolve_overlap(self, frame, handmask, threshold=0.75):
        overlap = cv.bitwise_and(handmask, handmask, mask=self.mask)
        if np.sum(overlap) / np.sum(self.mask) > threshold:
            self.change(frame, self.param, self.amount_change)
        elif self.persist:
            # no change, but persist the effect
            self.change(frame, self.param, 0)

class FrameManager:
    @classmethod
    def get_aspect_ratio(cls, frame):
        h,w = frame.shape[:2]
        return h/w

    @classmethod
    def adjust(cls, frame, tar_sz):
        frame = cv.resize(frame, tar_sz)
        frame = cv.flip(frame, 1)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        return frame

    @classmethod
    def preprocess(cls, frame, sz, device=None, half=False):
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


def main():    
    # initiate video stream
    vc = cv.VideoCapture(0)
    time.sleep(2)

    # initiate parameters controled via `buttons`
    params = {
        "width": Param("width", 640, 200, 1000, resize_frame),
        "blur": Param("blur", 0, 0, 20, blur_faces)
    }

    buttons = [
        Button("winincrease", "imgs/buttons/win_plus.png", params["width"], 1),
        Button("windecrease", "imgs/buttons/win_minus.png", params["width"], -1),
        Button("more_blur", "imgs/buttons/blur_plus.png", params["blur"], 1, persist=True),
        Button("less_blur", "imgs/buttons/blur_minus.png", params["blur"], -1, persist=True)
    ]

    while True:
        # read video frame
        _, frame = vc.read()
        aspect_ratio = FrameManager.get_aspect_ratio(frame)
    
        # resize and flip the frame + fix channel order (cv default: BGR)
        frame = FrameManager.adjust(
            frame, (params["width"].val, int(params["width"].val * aspect_ratio))
        )

        # convert to correct format for tf 
        input_tensor = FrameManager.preprocess(
            frame, (IMGSZ, IMGSZ), device=DEVICE, half=half
        )
    
        # get hand detections for current frame
        #(will always make 100 detections sorted by object probability score)
        hand_detections = hand_detector(input_tensor)[0]

        # get hand masks
        handmask = get_hand_masks(
            hand_detections, 
            frame, 
            threshold=0.7,
            show_bbox=int(args["show_hand_bbox"]),
            show_mask=int(args["show_hand_mask"])
        )

        # add buttons to frame and resolve possible overlaps
        for btn in buttons:
            btn.add(frame)
            btn.resolve_overlap(frame, handmask)

        # display final frame
        cv.imshow("press 'q' to quit",  frame[:,:,::-1])
        
        stopkey = cv.waitKey(1)
        if stopkey == ord("q"):
            break

    vc.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    args = get_args()

    # adjust float precision: if no cuda - use float32
    half = False if DEVICE.type == 'cpu' else HALF

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