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
    res = [250, 180] # x,y: screen size (resolution)
    center = [res[1]//2, res[0]//2] # pattern center
    c = np.array(center).reshape(2,1)
    dim = 180//6 # reference size, all polygons will be scaled against it 
    dim0 = dim # record the initial dim for future reference
    
    # general
    pf = PerlinFlow()
    perlin = pf.get_perlin()

    # triangles
    n_tri = 3   
    tri_rad = 0.7 * dim; # center of the circle that touches triangle vetices
    tri_num = 10 # 10 points per each edge
    perlin_ids = [i * perlin.shape[0]//n_tri for i in range(n_tri)]
    tris = [Polygon(3, center, tri_rad, perlin=perlin, perlin_idx=perlin_ids[i]) 
            for i in range(n_tri)]

    # squares
    n_sq = 6
    sq_rad = 1.1 * dim
    perlin_ids = [np.random.choice(perlin.shape[0])] * n_sq
    squares = [Polygon(4, center, (0.93 if i%2 else 1)*sq_rad, 
                       perlin=perlin, perlin_idx=perlin_ids[i]) 
               for i in range(n_sq)]
    for i in range(0,n_sq,2):
        squares[i  ].rotate_vertices(i//2 * 2*np.pi/(n_sq//2) / 4)
        squares[i+1].rotate_vertices(i//2 * 2*np.pi/(n_sq//2) / 4)

    # hex
    n_hex = 2
    hex_rad = 1.40 * dim
    perlin_ids = [i * perlin.shape[0] // (n_hex//2) 
                  for i in range(n_hex//2) for _ in range(2)]
    hexes   = [Polygon(6, center, (0.93 if i%2 else 1)*hex_rad, 
                       perlin=perlin, perlin_idx=perlin_ids[i]) 
              for i in range(n_hex)]

    # almost circle
    n_circ = 1
    circ_rad = [i * dim for i in [1.15, 1.40, 1.45]]
    circs = [Polygon(18, center, circ_rad[i], perlin=perlin) 
             for i in range(n_circ)]
    
    # mini triangles
    n_mtri = 6
    mtri_rad = 0.2*dim
    mtri_dist = 1.55 * dim
    mtris = [Polygon(3, center, mtri_rad, 
                     dist_from_center=mtri_dist, perlin=perlin)
             for i in range(n_mtri)]
    for i,mtri in enumerate(mtris):
        mtri.rotate_vertices(i * 2*np.pi/n_mtri)  
    
    # mini circles in the center
    n_mcirc = 3
    mcirc_rad = 0.4*dim
    mcirc_dist = 0.3*dim
    perlin_idx = np.random.choice(perlin.shape[0])
    mcircs = [Polygon(12, center, mcirc_rad, 
                      dist_from_center=mcirc_dist, 
                      perlin=perlin,
                      perlin_idx=perlin_idx)
              for i in range(n_mcirc)]
    for i,mcirc in enumerate(mcircs):
        mcirc.rotate_vertices(i * 2*np.pi/n_mcirc) 
    
    # pent
    n_pent = 1
    pent_rad = 0.3*dim
    pent_dist = 0
    perlin_idx = np.random.choice(perlin.shape[0])
    pents = [Polygon(5, center, pent_rad, 
                     perlin=perlin, 
                     perlin_idx=perlin_idx)]    
    for pent in pents:
        pent.initialize_vertices(angles=[2*i * 2*np.pi/5 for i in range(6)])
        pent.initialize_dists()
        
    # mpents
    n_mpent = 10
    mpent_rad = 0.3 * dim
    mpent_dist = 1.3 * dim
    perlin_idx = np.random.choice(perlin.shape[0])
    mpents = [Polygon(5, center, pent_rad, dist_from_center=mpent_dist,
                      perlin=perlin, 
                      perlin_idx=perlin_idx)
              for i in range(n_mpent)]
    for i,mpent in enumerate(mpents):
        #mpent.vertices -= (mpent.center + mpent.c2p)        
        mpent.initialize_vertices(angles=[2*i * 2*np.pi/5 for i in range(6)])
        mpent.initialize_dists()
        mpent.rotate_vertices(i * 2*np.pi/n_mpent)
        
    # red
    vertices = get_red() * dim #+ c.reshape(1,2)
    reds = [Pattern(vertices, center, dist_from_center=0)]

    
    # combine polygons into one pattern
    #pattern = pents + tris + squares + hexes + circs + mtris #+ mcircs #
    pattern =  mpents + reds
    
    # keep updating the pattern
    time = 50
    for t in range(time):
        center[0] += 0.5*np.cos(0.01*t) 
        center[1] += 2*np.cos(0.03*t) * np.sin(0.5*t)
        dim = dim0 * (0.25*np.cos(0.5*t) + 1)
        
        plt.clf()
        
        scale = dim/dim0 
        for fig in pattern:            
            fig.update_vertices(center=center, scale=scale)
            plt.plot(fig.vertices.T[1], fig.vertices.T[0])
        
        plt.scatter(center[1], center[0])    
       
        plt.title(f'{scale}')
        plt.xlim([0, res[0]])
        plt.ylim([0, res[1]])
        plt.pause(0.01) 
        

        
        
     
