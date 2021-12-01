#!/usr/bin/env python3
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