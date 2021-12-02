#!/usr/bin/env python3
from types import FunctionType
from typing import List, Tuple, Dict, Set, Optional, Union

import numpy as np
import cv2 as cv

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
        name: str, 
        val: Union[int,float], 
        min_val: Union[int,float], 
        max_val: Union[int,float], 
        effect_fun: FunctionType
    ):
        self.name = name
        self.val = val
        self.min_val = min_val
        self.max_val = max_val
        self.effect_fun = effect_fun

    def update(
        self, 
        frame: np.ndarray, 
        param: 'Param', 
        amount_change: Union[int, float]
    ) -> None:
        """
        this function should be `subscribed` to 
        `Button`'s events from observable `Button`;
        not subscribing it a param instantiation
        because single param observes several buttons,
        so it's easier to arrange it from button's side
        """
        if param == self:
            self.val += amount_change
            self.val = np.clip(self.val, self.min_val, self.max_val)
            self.effect_fun(frame, self.val)

class Button(Observable):
    def __init__(
        self, 
        name: str, 
        path: str, 
        param: Param, 
        amount_change: Union[int, float], 
        persist: bool = False
    ):
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

    def load(self) -> None:
        """
        loads button from provided image path
        and stores button image and mask refence
        """
        comb = cv.imread(self.path, cv.IMREAD_UNCHANGED)
        comb = cv.cvtColor(comb, cv.COLOR_BGRA2RGBA)
        self.btn_ref = comb[:,:,:3]
        self.mask_ref = comb[:,:,3]
        self.btn = self.btn_ref.copy()
        self.mask = self.mask_ref.copy()

    def add(self, frame: np.ndarray) -> None:
        """
        adds `this` button to specified `frame`
        """
        # resize if needed
        h,w,c = frame.shape
        if self.mask != (h,w):
            self.btn = cv.resize(self.btn_ref, (w,h))
            self.mask = cv.resize(self.mask_ref, (w,h))

        fg = cv.bitwise_and(self.btn, self.btn, mask=self.mask)
        bg = cv.bitwise_and(frame, frame, mask=cv.bitwise_not(self.mask))
        cv.add(bg, fg, dst=frame)

    def resolve_overlap(
        self, 
        frame: np.ndarray, 
        mask: np.ndarray, 
        threshold: float = 0.75
    ) -> None:
        """
        checks if `this` button overlaps with `mask`;
        if overlap to button-mask ratio is above `threshold`
        triggers button effect;
        if this ratio is below `threshold` but `persist` flag
        is set to `True`, still triggers the effect function
        but doesn't modify value of triggered observer
        """
        overlap = cv.bitwise_and(mask, mask, mask=self.mask)
        if np.sum(overlap) / np.sum(self.mask) > threshold:
            self.change(frame, self.param, self.amount_change)
        elif self.persist:
            # no change, but persist the effect
            self.change(frame, self.param, 0)