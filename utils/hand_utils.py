#!/usr/bin/env python3
"""
Created on Sun Aug 18 12:49:29 2019

@author: axeh
"""

import numpy as np
import cv2 as cv
import imutils

class ButtonOld:
    def __init__(self, center, rad):
        self.c = center
        self.r = rad
        self.action = lambda p: p


def sample_hand(handbox, handmask):
    # mask no-hand pixels (no hand = True)
    if np.max(handmask) == 255:
        handmask  = handmask <= 230

    handhsv = cv.cvtColor(handbox, cv.COLOR_BGR2HSV)

    # get all non-masked H and S values 
    allh = np.array([handhsv[i,j,0] for i in range(handhsv.shape[0])
                           for j in range(handhsv.shape[1])
                           if not handmask[i,j,0]])
    alls = np.array([handhsv[i,j,1] for i in range(handhsv.shape[0])
                           for j in range(handhsv.shape[1])
                           if not handmask[i,j,0]])
    # get min/max values along H and S dimension
    delta_up, delta_down = 20,80
    hmin = max(0,   allh.mean() - 2*allh.std() - delta_down)
    hmax = min(255, allh.mean() + 2*allh.std() + delta_up)
    smin = max(0,   alls.mean() - 2*alls.std() - delta_down)
    smax = min(255, alls.mean() + 2*alls.std() + delta_up)
    
    lower,upper = (hmin,smin,0),(hmax,smax,200)
    return lower, upper

def add_buttons(frame):
    "params = [width, blur]"
    h,w = frame.shape[:2]
    r = int(0.05*h)
    buttons = dict()

    c0 = int(0.9*w), int(0.8*h)
    cv.circle(frame, c0, r, (0,0,255), -1)
    cv.putText(frame, "+", c0, cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    buttons["inflate"] = ButtonOld(c0, r)
    buttons["inflate"].action = lambda p: [p[0]+1] + p[1:]

    c1 = int(0.1*w), int(0.8*h)
    cv.circle(frame, c1, r, (255,0,0), -1)
    cv.putText(frame, "-", c1, cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    buttons["shrink"] = ButtonOld(c1, r)
    buttons["shrink"].action = lambda p: [p[0]-1] + p[1:]

    c2 = int(0.9*w), int(0.6*h)
    cv.circle(frame, c2, r, (100,0,230), -1)
    cv.putText(frame, "+blur", c2, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)
    buttons["more blur"] = ButtonOld(c2, r)
    buttons["more blur"].action = lambda p: [p[0]] + [p[1]+1] + p[2:]

    c3 = int(0.1*w), int(0.6*h)
    cv.circle(frame, c3, r, (230,0,100), -1)
    cv.putText(frame, "-blur", c3, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)
    buttons["less blur"] = ButtonOld(c3, r)
    buttons["less blur"].action = lambda p: [p[0]] + [p[1]-1] + p[2:]

    return frame, buttons

def get_button_masks(frame, buttons):
    sz = frame.shape

    masks = dict()
    for button in buttons.keys():
        mask = np.zeros(sz[:2], dtype="uint8")
        mask = cv.circle(mask, buttons[button].c, buttons[button].r,
                        (255,255,255), -1)
        masks[button] = mask

    return masks

def get_foreground_mask(frame, frameref):
    if len(frameref.shape) == 2 and len(frame.shape) == 3:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    delta = np.abs(frame - frameref)

    foreground_mask = np.zeros(delta.shape)
    foreground_mask[delta > 30] = 255
    return foreground_mask.astype("uint8")

class Button:
    def __init__(self, path):
        self.action = lambda p: p
        self.path = path
        
    def load_button_and_mask(self):
        comb = cv.imread(self.path, cv.IMREAD_UNCHANGED)
        comb = cv.cvtColor(comb, cv.COLOR_BGRA2RGBA)
        self.btn_ref = comb[:,:,:3]
        self.mask_ref = comb[:,:,3]
        self.btn = self.btn_ref.copy()
        self.mask = self.mask_ref.copy()

def mk_buttons():
    buttons = dict()
    
    buttons["inflate"] = Button("imgs/buttons/win_plus.png")
    buttons["inflate"].action = lambda p: [p[0]+1] + p[1:]
    buttons["inflate"].load_button_and_mask()
    
    buttons["shrink"] = Button("imgs/buttons/win_minus.png")
    buttons["shrink"].action = lambda p: [p[0]-1] + p[1:]
    buttons["shrink"].load_button_and_mask()
    
    buttons["more blur"] = Button("imgs/buttons/blur_plus.png")
    buttons["more blur"].action = lambda p: [p[0]] + [p[1]+1] + p[2:]
    buttons["more blur"].load_button_and_mask()

    buttons["less blur"] = Button("imgs/buttons/blur_minus.png")
    buttons["less blur"].action = lambda p: [p[0]] + [p[1]-1] + p[2:]
    buttons["less blur"].load_button_and_mask()
    
    return buttons
    
def add_buttons_test(frame, buttons):
    h,w,c = frame.shape
    
    masks = np.zeros((h,w), dtype=np.uint8)
    fg = np.zeros((h,w,c), dtype=np.uint8)
    for button in buttons.keys():
        if buttons[button].mask.shape != (h,w):
            buttons[button].mask = cv.resize(buttons[button].mask_ref, (w,h))
            buttons[button].btn = cv.resize(buttons[button].btn_ref, (w,h))
                    
        masks += buttons[button].mask
        fg += cv.bitwise_and(buttons[button].btn, buttons[button].btn, 
                             mask=buttons[button].mask)
        
    bg = cv.bitwise_and(frame, frame, mask=cv.bitwise_not(masks))
    cv.add(bg, fg, dst=frame)

def blur_box(frame, pt1, pt2, k, sigma, weight=0.7):
    xmin,ymin = pt1
    xmax,ymax = pt2

    blur     = frame.copy()
    blur     = cv.GaussianBlur(blur, (k,k), sigma)

    frame[ymin:ymax,xmin:xmax,:] = blur[ymin:ymax,xmin:xmax,:]

    return frame
