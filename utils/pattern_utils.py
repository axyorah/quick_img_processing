# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 22:03:36 2020

@author: elvir
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.perlin_flow import PerlinFlow


def rotate(vecs: np.ndarray, angle: float):
    """
    rotate (1,2) vector or (N,2) mtx (N vectors) by given angle around origin
    INPUTS:
        vecs: (N,2) array of vectors to be rotated stacked along the 0-axis
        angle: float: rotation angle in radians
    """
    return np.dot(vecs, 
                  np.array([[np.cos(angle), np.cos(np.pi/2 + angle)],
                            [np.sin(angle), np.sin(np.pi/2 + angle)]]))
    
class Pattern:
    def __init__(self, vertices: np.ndarray, 
                       center: list or np.ndarray, 
                       dist_from_center=0,
                       perlin=None,
                       perlin_idx=0):
        self.vertices = vertices # (N,2), for closed figures 1st and last rows are the same
        self.center = np.array(center) # coordinates of center [y,x]
        self.delta_center = np.array([[0,0]]) # displacement of the pattern's center compared to the prev setp
                
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
        self.dists0 = np.linalg.norm(self.vertices - self.center, axis=1, keepdims=True)
        self.dists = self.dists0.copy()
        
    def get_center_displacement(self, center=None):
        center = np.array(center if center is not None else self.center).\
                 reshape(1,2)
        self.delta_center = center - self.center
        self.center = center
        return self.delta_center
        
    def adjust_scale(self, scale=None):
        if scale is not None:  
            self.dists = np.linalg.norm(self.vertices - self.center, 
                                        axis=1, keepdims=True)
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
        self.center = np.array(center if center is not None else self.center)
        
        self.vertices = rotate(self.vertices-self.center, angle) + \
                        self.center
        return self.vertices.astype(int)
   
    def update_vertices(self, center=None, scale=None):
        self.frame += 1
        angle = 2*np.pi/100 + \
                1*self.perlin[self.perlin_idx, self.frame%self.perlin.shape[1]]
        
        self.adjust_scale(scale=scale)
        self.adjust_center(center=center)
                
        self.rotate_vertices(angle, center=center)
        return self.vertices.astype(int)
    
class Polygon(Pattern):
    def __init__(self, num_vertices: int,
                       center: list or np.ndarray,                        
                       rad: float,
                       dist_from_center=0,
                       perlin=None,
                       perlin_idx=0):        
        super().__init__(np.array([[0,0]]), # <-- ugly dummy vertices
                         center,
                         dist_from_center=dist_from_center,
                         perlin=perlin,
                         perlin_idx=perlin_idx)
        self.num_vertices = num_vertices
        self.rad = rad
        self.initialize_vertices() 
        self.initialize_dists() # <-- recalculate dists, since we passed dummies before
        
        
    def initialize_vertices(self, angles=None):
        if angles is None:
            angles = [2*np.pi/self.num_vertices * i 
                      for i in range(self.num_vertices)] + [0]

        self.vertices = np.concatenate([rotate(np.array([[0, self.rad]]), angle)
                                        for angle in angles],
                                       axis=0) + \
                        self.center + self.c2p
        
        return self.vertices
        

  
#%%    
if __name__ == '__main__':
    pass