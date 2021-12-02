#!/usr/bin/env python3
from typing import List, Tuple, Dict, Set, Optional, Union

import numpy as np
import torch
import cv2 as cv
import argparse
import time

from utils.detector_utils import Detector
from utils.hand_utils import Button, Param
from utils.yolo_utils_by_ultralytics.yolo import Model

FACE_FILTER_PROTO = "./dnn/facial/face_deploy.prototext"
FACE_FILTER_RCNN = "./dnn/facial/face_res10_300x300_ssd_iter_140000.caffemodel"
HAND_DETECTOR_YAML = "./dnn/yolov5s.yaml"
HAND_DETECTOR_DICT = "./dnn/yolov5_palm_state_dict.pt"

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
HALF = False

def get_args() -> Dict:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-b", "--show_hand_bbox", default="0",
        help=(
            "set to 1 to show hand bounding boxes "
            "and to 0 to ignore it (default)"
        )
    )

    parser.add_argument(
        "-t", '--threshold', type=float, default=0.7,
        help=(
            "probability threshold for hand detector; "
            "by default set to `0.7`; decrease it detector "
            "has difficulties detecting your hand "
            "(e.g., because of bad lighting conditions)"
        )
    )

    args = vars(parser.parse_args())
    args["show_hand_bbox"] = False if args["show_hand_bbox"] == "0" else True
    args["threshold"] = float(args["threshold"])

    return args


class YoloHandDetector(Detector):
    IMGSZ = 448
    def __init__(
        self, 
        config_path: str, 
        weights_path: str, 
        class_dict: Dict = None,
        device: torch.device = torch.device('cpu'),
        half: bool = False
    ):
        super().__init__(config_path, weights_path, class_dict)
        self.device = device
        self.half = half
        self.load()

    def load(self) -> Model:
        # restore model from state_dict checkpoint
        self.detector = Model(cfg=self.config_path, nc=len(self.class_dict))
        ckpt = torch.load(self.weights_path)
        self.detector.load_state_dict(ckpt['model_state_dict'])

        # add non-max suppression
        self.detector.nms()

        # adjust precision
        if self.half:
            self.detector.half()
        else:
            self.detector.float()

        # use cuda is available
        self.detector.to(self.device)

        # switch to inference mode
        self.detector.eval()
        return self.detector

    def preprocess(self, frame: np.ndarray) -> torch.TensorType:
        """
        preprocesses raw img frame (np.ndarray) for 
        yolo5 torch detector 
        """
        tensor = cv.resize(frame, (self.IMGSZ, self.IMGSZ))   
        tensor = tensor.transpose(2, 0, 1)
        tensor = torch.from_numpy(tensor)
        tensor = torch.unsqueeze(tensor, 0)
        tensor = tensor.half() if self.half else tensor.float()
        tensor = tensor / 255.0
        tensor = tensor.to(self.device)

        return tensor

    def detect(
        self, 
        input_tensor: torch.TensorType, 
        threshold: float = 0.75,
        sz: Tuple[int,int] = (None, None)
    ) -> List[List[float]]:
        """
        runs yolov5 on preprocessed `input_tensor` 
        and stores hand detections and masks
        with class probability above `threshold`; 
        stores detections in format:
        ```
        [
            [x1, y1, x2, y2, probability, class_idx],
            ...
        ]
        ```
        masks are stored as 2D np arrays of 0's
        with area within object bboxes set to 255's;
        if `sz` is specified as (tar height, tar width)
        detections will be adjusted to match provided shape;
        returns detections;
        detections and masks can be assessed as:
        ```
        detector = YoloHandDetector(...)
        detector.detect(...)
        detections = detector.detections
        mask = detector.mask
        ```
        """
        h, w = sz or input_tensor.shape[:2]
        raw_detections = self.detector(input_tensor)[0]

        self._detections = []
        self._mask = np.zeros((h, w), dtype=int)

        # raw output contains 100 detections sorted by obj prob score
        for x1, y1, x2, y2, prob, clss in raw_detections:
            if prob < threshold:
                continue
        
            # get box coordinates
            x1, x2 = map(lambda x: int(x * w / self.IMGSZ), [x1, x2])
            y1, y2 = map(lambda y: int(y * h / self.IMGSZ), [y1, y2])
            clss = int(clss)

            self._detections.append([x1, y1, x2, y2, prob, clss])
            self._mask[y1:y2, x1:x2] = 255

        return self._detections

    def show(self, frame: np.ndarray) -> None:
        for x1, y1, x2, y2, prob, clss in self._detections:        
            # get box coordinates
            cv.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)


class CvFaceDetector(Detector):
    IMGSZ = 300
    def __init__(
        self, 
        config_path: str, 
        weights_path: str, 
        class_dict: Dict = None
    ):
        super().__init__(config_path, weights_path, class_dict)
        self.load()
        
    def load(self) -> cv.dnn_Net:
        self.detector = cv.dnn.readNetFromCaffe(
            self.config_path,
            self.weights_path
        )
        return self.detector

    def preprocess(self, frame: np.ndarray):
        return cv.dnn.blobFromImage(
            cv.resize(frame, (self.IMGSZ, self.IMGSZ)), 
            1.0, 
            (self.IMGSZ, self.IMGSZ), 
            (104.0, 177.0, 123.0)
        )

    def detect(
        self, 
        blob: np.ndarray, 
        threshold: float = 0.5, 
        sz: Tuple[int,int] = (None, None)
    ) -> List[List[int]]:
        """
        runs opencv built-in face detector on preprocessed 
        `blob` (numpy array with shape (1,3,300,300));
        stores detections with class probability above
        `threshold` in format:
        ```
        [
            [x1, y1, x2, y2, probability, class_index],
            ...
        ]
        ```
        if `sz` is specified as (tar height, tar width)
        detections will be adjusted to match provided shape;
        returns detections
        """
        h, w = sz or blob.shape[-2:]

        self.detector.setInput(blob)
        raw_detections = self.detector.forward()

        detections = []
        for i in range(raw_detections.shape[2]):
            p,x1,y1,x2,y2 = raw_detections[0,0,i,2:7]

            if p < threshold:
                continue

            x1,x2 = map(lambda x: int(x * w), [x1,x2])
            y1,y2 = map(lambda y: int(y * h), [y1,y2])

            detections.append([x1, y1, x2, y2, p, 0])

        return detections


def blur_box(
    frame: np.ndarray, 
    pt1: Union[List[float], np.ndarray], 
    pt2: Union[List[float], np.ndarray], 
    k: int, 
    sigma: int
) -> np.ndarray:

    xmin,ymin = pt1
    xmax,ymax = pt2

    blur     = frame.copy()
    blur     = cv.GaussianBlur(blur, (k,k), sigma)

    frame[ymin:ymax,xmin:xmax,:] = blur[ymin:ymax,xmin:xmax,:]

    return frame

def blur_faces(frame: np.ndarray, blur: int) -> np.ndarray:
    """
    applies gaussian blur to all faces detected in `frame`;
    derives gaussian params `k` and `sigma` from specified `blur`:
    ```
    k = 1 + 2*blur
    sigma = 1 + blur
    ```
    """
    blob = face_detector.preprocess(frame)
    detections = face_detector.detect(blob, sz=frame.shape[:2])

    k     = 1 + 2*blur
    sigma = 1 +   blur
    
    for x1, y1, x2, y2, prob, clss in detections:
        blur_box(frame, (x1,y1), (x2,y2), k, sigma)
        
    return frame
        
def resize_frame(frame: np.ndarray, width: Union[int,float]):
    """
    resizes current `frame` to specified `width` 
    keeping the aspect ratio;
    returns resized frame
    """
    width = int(width)
    height = int(FrameManager.get_aspect_ratio(frame) * width)
    return cv.resize(frame, (width, height))

class FrameManager:
    @classmethod
    def get_aspect_ratio(cls, frame: np.ndarray) -> float:
        """aspect ratio is ratio of img height to img width"""
        h,w = frame.shape[:2]
        return h/w

    @classmethod
    def adjust(cls, frame: np.ndarray, tar_sz: Tuple[int,int]) -> np.ndarray:
        """resizes and flips image frame"""
        frame = cv.resize(frame, tar_sz)
        frame = cv.flip(frame, 1)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        return frame


def main():
    # get command line args
    args = get_args()

    # initiate parameters controled via `buttons`
    params = {
        "width": Param("width", 640, 200, 1000, resize_frame),
        "blur": Param("blur", 0, 0, 20, blur_faces)
    }

    buttons = [
        Button("win_plus", "imgs/buttons/win_plus.png", params["width"], 1),
        Button("win_minus", "imgs/buttons/win_minus.png", params["width"], -1),
        Button("blur_plus", "imgs/buttons/blur_plus.png", params["blur"], 1, persist=True),
        Button("blur_minus", "imgs/buttons/blur_minus.png", params["blur"], -1, persist=True)
    ]

    # initiate video stream
    vc = cv.VideoCapture(0)
    time.sleep(2)
    while True:
        # read video frame
        _, frame = vc.read()
        aspect_ratio = FrameManager.get_aspect_ratio(frame)
    
        # resize and flip the frame + fix channel order (cv default: BGR)
        frame = FrameManager.adjust(
            frame, (params["width"].val, int(params["width"].val * aspect_ratio))
        )

        # find hand bboxes and masks
        input_tensor = hand_detector.preprocess(frame)
        hand_detector.detect(
            input_tensor, 
            threshold=args["threshold"],
            sz=frame.shape[:2]
        )

        # add buttons to frame and resolve possible overlaps
        for btn in buttons:
            btn.add(frame)
            btn.resolve_overlap(frame, hand_detector.mask)

        # display final frame
        if args["show_hand_bbox"]:
            hand_detector.show(frame)

        cv.imshow("press 'q' to quit",  frame[:,:,::-1])
        
        stopkey = cv.waitKey(1)
        if stopkey == ord("q"):
            break

    vc.release()
    cv.destroyAllWindows()

if __name__ == "__main__":

    face_detector = CvFaceDetector(
        FACE_FILTER_PROTO, 
        FACE_FILTER_RCNN, 
        class_dict={0: 'face'}
    )

    hand_detector = YoloHandDetector(
        HAND_DETECTOR_YAML,
        HAND_DETECTOR_DICT,
        class_dict={0: 'hand'}, 
        device=DEVICE, 
        half=False if DEVICE.type == 'cpu' else HALF
    )

    main()