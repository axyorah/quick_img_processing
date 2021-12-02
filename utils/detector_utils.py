#!/usr/bin/env python3
from typing import List, Tuple, Dict, Optional, Union
from abc import ABC

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
