#!/usr/bin/env python3

import numpy as np
import cv2 as cv
#import imutils

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
    
def add_buttons(frame, buttons):
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
