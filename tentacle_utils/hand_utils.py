# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 12:49:29 2019

@author: axeh
"""

import numpy as np
import cv2 as cv
import imutils

class Button:
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
    buttons["inflate"] = Button(c0, r)
    buttons["inflate"].action = lambda p: [p[0]+1] + p[1:]
        
    c1 = int(0.1*w), int(0.8*h)    
    cv.circle(frame, c1, r, (255,0,0), -1)
    cv.putText(frame, "-", c1, cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    buttons["shrink"] = Button(c1, r)
    buttons["shrink"].action = lambda p: [p[0]-1] + p[1:]
    
    c2 = int(0.9*w), int(0.6*h)    
    cv.circle(frame, c2, r, (100,0,230), -1)
    cv.putText(frame, "+blur", c2, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)
    buttons["more blur"] = Button(c2, r)
    buttons["more blur"].action = lambda p: [p[0]] + [p[1]+1] + p[2:]
    
    c3 = int(0.1*w), int(0.6*h)    
    cv.circle(frame, c3, r, (230,0,100), -1)
    cv.putText(frame, "-blur", c3, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)
    buttons["less blur"] = Button(c3, r)
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

def blur_box(frame, pt1, pt2, k, sigma, weight=0.7):
    xmin,ymin = pt1
    xmax,ymax = pt2       
    
    blur     = frame.copy()
    blur     = cv.GaussianBlur(blur, (k,k), sigma)
    
    frame[ymin:ymax,xmin:xmax,:] = blur[ymin:ymax,xmin:xmax,:]
    
    return frame