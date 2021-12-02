#!/usr/bin/env python3
from typing import List, Tuple, Dict, Optional, Union
from abc import ABC

import numpy as np
import cv2 as cv
import torch
from utils.yolo_utils_by_ultralytics.yolo import Model


class Detector(ABC):
    def __init__(
        self, 
        config_path: str, 
        weights_path: str, 
        class_dict: Dict = None
    ):
        self.config_path = config_path
        self.weights_path = weights_path
        self.class_dict = class_dict or {0: 'obj'}
        self.detector = None
        self._detections = None
        self._mask = None

    def load(self):
        pass

    def preprocess(cls, frame):
        pass

    def detect(self, frame):
        pass

    @property
    def detections(self):
        return self._detections

    @property
    def mask(self):
        return self._mask


class YoloTorchDetector(Detector):
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


class CvObjectDetector(Detector):
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
