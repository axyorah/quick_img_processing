# -*- coding: utf-8 -*-
from typing import List, Tuple, Dict, Set, Optional, Union
import os
import cv2 as cv
import numpy as np
from numpy.lib.twodim_base import mask_indices

if __name__ == "__main__":
    from perlin_flow import PerlinFlow
else:
    from utils.perlin_flow import PerlinFlow

def rotate(vecs: np.ndarray, angle: float):
    """
    rotate (1,2) vector or (N,2) mtx (N vectors) by given angle around origin
    INPUTS:
        vecs: (N,2) array of vectors to be rotated stacked along the 0-axis
        angle: float: rotation angle in radians
    """
    return np.dot(
        vecs, 
        np.array([
            [np.cos(angle), np.cos(np.pi/2 + angle)],
            [np.sin(angle), np.sin(np.pi/2 + angle)]
        ])
    )

class Shape:
    def __init__(self, vertices):
        self._center = np.array([[0, 0]], dtype=float)
        self._vertices = np.array(vertices).reshape(-1,2)
        self._dist0 = self._get_avg_dist_from_center()
        self._scale0 = 1
        
    def _get_avg_dist_from_center(self):
        self._dist = np.linalg.norm(self._vertices, axis=1).mean()
        return self._dist        
        
    def translate(self, vec):
        self._center += np.array(vec, dtype=float).reshape(-1, 2)
        return self
    
    def scale(self, scalar, absolute=True):
        if absolute:
            coeff = self._dist0 / self._get_avg_dist_from_center()
        else:
            coeff = 1
        self._vertices *= (scalar * coeff)
        return self
    
    def rotate(self, angle):
        self._vertices = rotate(self._vertices, angle)
        return self
    
    @property
    def vertices(self):
        return self._vertices + self._center


class Poly(Shape):
    def __init__(self, num_vertices, angle=None):
        self.num_vertices = num_vertices
        self.angle = angle
        self._get_init_vertices()            
        super().__init__( self._vertices )

    def _get_init_vertices(self):
        angle = self.angle or 2*np.pi/self.num_vertices
        angles = [
            angle * i for i in range(self.num_vertices + 1)
        ]

        self._vertices = np.concatenate([
            rotate(np.array([[1, 0]]), angle)
            for angle in angles
        ], axis=0) 
        
        return self._vertices


class PerlinShape(Shape):
    def __init__(self, vertices, perlin=None, perlin_modifier=1, perlin_row_idx=None):
        super().__init__(vertices)
        self.perlin = perlin if perlin is not None else PerlinFlow().get_perlin()
        self.perlin_modifier = perlin_modifier
        self.perlin_col_idx = 0
        self.perlin_row_idx = (
            perlin_row_idx if perlin_row_idx is not None
            else np.random.choice(self.perlin.shape[0])
        ) 
        
    def next(self):
        self.perlin_col_idx += 1
        self.perlin_col_idx %= self.perlin.shape[1]

        self.rotate(
            self.perlin[
                self.perlin_row_idx, 
                self.perlin_col_idx
            ] * self.perlin_modifier 
        )
        return self


class PerlinComplexShape:
    def __init__(self, perlin=None):
        self.perlin = perlin if perlin is not None else PerlinFlow().get_perlin()
        self.children = [] # each child is `PerlinShape` with `next()` method

    def translate(self, pt1, pt2):
        """translate into the center of the box between pt1 and pt2"""
        x1,y1 = pt1
        x2,y2 = pt2
        
        center = np.array([[(x2+x1)//2, (y2+y1)//2]])
        
        for child in self.children:
            child._center = center

        return self
    
    def scale(self, pt1, pt2):
        x1,y1 = pt1
        x2,y2 = pt2
        
        scale = max((x2-x1),(y2-y1))

        for child in self.children:
            child.scale(scale * self._scale0)

        return self
    
    def next(self):
        for child in self.children:
            child.next()

        return self
    
    @property
    def vertices(self):
        return [child.vertices for child in self.children]

    def draw(self, frame):
        for child in self.children:
            cv.fillPoly(frame, [child.vertices.astype(int)], (0,0,255))


class FrameSqeuence:
    PATH_FRAMES_DIR = ''
    def __init__(self):
        self._isongoing = False
        self._ongoingframe = 0
        self.frames = []
        self.masks = []
        self.fade = 5 # num frames that gradually fade at the end

    def load(self):
        for name in os.listdir(self.PATH_FRAMES_DIR):
            fname = os.path.join(self.PATH_FRAMES_DIR, name)
            if not os.path.isfile(fname):
                continue

            combined = cv.imread(fname, cv.IMREAD_UNCHANGED)
            self.frames.append(combined[:,:,:3])
            if combined.shape[2] > 3:
                self.masks.append(combined[:,:,3])

    @property
    def isongoing(self):
        return self._isongoing

    def maybe_begin(self):
        if self.isongoing:
            return
        else:
            self._isongoing = True
            self._ongoingframe = 0

    def maybe_draw(self, frame, pt1, pt2):
        if not self.isongoing:
            return
        if self._ongoingframe >= len(self.frames):
            self._isongoing = False
            self._ongoingframe = 0
            return 

        kaboom = self.frames[self._ongoingframe]
        mask = (
            self.masks[self._ongoingframe] if self.masks
            else 255 * np.ones(frame.shape[:2], dtype=int)
        )

        fadedegree = max(
            0, 
            self.fade - (len(self.frames) - self._ongoingframe) + 1
        ) / self.fade 

        combined = frame.copy()
        cv.copyTo(kaboom, mask, combined)
        cv.addWeighted(
            combined, 1 - fadedegree, frame, fadedegree, 0, frame
        )

        self._ongoingframe += 1

        










    
class Pattern:
    def __init__(
        self, 
        vertices: np.ndarray, 
        center: Union[List,np.ndarray], 
        dist_from_center: float = 0,
        perlin: Optional[np.ndarray] = None,
        perlin_idx: int = 0
    ):
        self.vertices = vertices # (N,2), for closed figures 1st and last rows are the same
        self.center = np.array(center) # coordinates of center [y,x]
        self.delta_center = np.array([[0,0]]) # displacement of the pattern's center compared to the prev step
                
        self.dist = dist_from_center        
        self.c2p = np.array([0,1]) * self.dist # center-to-point vector
        self.shift_vertices() 
        self.initialize_dists() # should follow `shift_vertices` (because of c2p)
        
        self.frame = 0
        if perlin is not None:
            self.perlin = perlin
        else:
            pf = PerlinFlow()
            self.perlin = pf.get_perlin()
        self.perlin_idx = perlin_idx
        
    def shift_vertices(self):
        self.vertices = self.vertices + self.center + self.c2p
        
    def initialize_dists(self):
        self.dists0 = np.linalg.norm(
            self.vertices - self.center, 
            axis=1, 
            keepdims=True
        )
        self.dists = self.dists0.copy()
        
    def get_center_displacement(self, center=None):
        center = np.array(
            center if center is not None else self.center
        ).reshape(1,2)
        self.delta_center = center - self.center
        self.center = center
        return self.delta_center
        
    def adjust_scale(self, scale=None):
        if scale is not None:  
            self.dists = np.linalg.norm(
                self.vertices - self.center, 
                axis=1, 
                keepdims=True
            )
            scale_relative = scale * self.dists0 / self.dists
            self.vertices = self.center + \
                (self.vertices - self.center) * scale_relative 
                
        return self.vertices  

    def adjust_center(self, center=None):
        # center around the new 'center'
        self.get_center_displacement(center=center)
        self.vertices += self.delta_center
        return self.vertices     
    
    def rotate_vertices(self, angle, center=None):
        self.center = np.array(
            center if center is not None else self.center
        )
        
        self.vertices = rotate(
            self.vertices-self.center, angle
        ) + self.center

        return self.vertices.astype(int)
   
    def update_vertices(self, center=None, scale=None):
        self.frame += 1
        angle = 2*np.pi/100 + \
                1*self.perlin[
                    self.perlin_idx, 
                    self.frame%self.perlin.shape[1]
                ]
        
        self.adjust_scale(scale=scale)
        self.adjust_center(center=center)
                
        self.rotate_vertices(angle, center=center)
        return self.vertices.astype(int)
    
class Polygon(Pattern):
    def __init__(
        self, 
        num_vertices: int,
        center: Union[List,np.ndarray],                        
        rad: float,
        dist_from_center: float = 0,
        perlin: Optional[np.ndarray] = None,
        perlin_idx: int = 0
    ):        
        super().__init__(
            np.array([[0,0]]), # <-- ugly dummy vertices
            center,
            dist_from_center=dist_from_center,
            perlin=perlin,
            perlin_idx=perlin_idx
        )
        self.num_vertices = num_vertices
        self.rad = rad
        self.initialize_vertices() 
        self.initialize_dists() # <-- recalculate dists, since we passed dummies before
        
        
    def initialize_vertices(self, angles=None):
        if angles is None:
            angles = [
                2*np.pi/self.num_vertices * i 
                for i in range(self.num_vertices)
            ] + [0]

        self.vertices = np.concatenate([
            rotate(np.array([[0, self.rad]]), angle)
            for angle in angles
        ], axis=0) + self.center + self.c2p
        
        return self.vertices
