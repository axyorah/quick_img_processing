#!/usr/bin/env python3
"""
Created on Sun Dec 17 16:43:56 2017

makes *circular* 3d perlin noise (surface) and rolls it over  
to give false impression that it's changing in time...
adds some cheap harmonics to make the "changing in time" effect stronger
(without this addition the repetitiveness of the rolling pattern 
gives away its 3d-ness...)

perlin tutorial: https://eev.ee/blog/2016/05/29/perlin-noise/

@author: axeh
"""

import numpy as np
import matplotlib.pyplot as plt

class PerlinFlow:
    def __init__(self,ver_grid=3, hor_grid=5, 
                      num_octaves=2, points_at_last_octave=4, circular=True):
        self.circular = circular
        self.ver_grid = ver_grid
        self.hor_grid = hor_grid
        self.num_octaves = num_octaves
        self.points_at_last_octave = points_at_last_octave
        
        self.n_ver = self.ver_grid*self.points_at_last_octave**self.num_octaves+1
        self.n_hor = self.hor_grid*self.points_at_last_octave**self.num_octaves+1
        
    def get_grid(self, raw=False):
        grid_list = np.meshgrid(np.linspace(0,self.hor_grid,self.n_hor),
                                np.linspace(0,self.ver_grid,self.n_ver)) 
        grid = np.concatenate((grid_list[0].reshape(self.n_ver,self.n_hor,1),
                               grid_list[1].reshape(self.n_ver,self.n_hor,1)),
                               axis=2)
        
        if raw == True: return grid
        else: return \
            grid[:(self.ver_grid-1)*self.points_at_last_octave**self.num_octaves,
                 :(self.hor_grid-1)*self.points_at_last_octave**self.num_octaves,:]
        
    def get_octaves(self):
        octaves_list = \
            [np.meshgrid(np.linspace(0,self.hor_grid,int((2**i) * self.hor_grid+1)),
                         np.linspace(0,self.ver_grid,int((2**i) * self.ver_grid+1)))
             for i in range(self.num_octaves)]

        octaves = \
            [np.concatenate((oc[0].reshape(oc[0].shape[0],oc[0].shape[1],1),
                             oc[1].reshape(oc[1].shape[0],oc[1].shape[1],1)),
                             axis=2)
             for oc in octaves_list]
        return octaves

    def smoothen(self,t):
        """ S-curve that is flat at 0 and 1:   6x⁵ - 15x⁴ + 10x³    """
        #return(np.minimum(np.maximum(3*t**2 - 2*t**3, 0),1))
        return np.minimum(np.maximum(6*t**5 - 15*t**4 + 10*t**3, 0),1)

    def get_masks(self,dx1,dx2):
        """returns 4 masks:
        masks are weight matrices 
        that assign '1' to one of the corners in the current cell,
        and '0' to other three corners;
        values of all other mtx elements are smoothly in-between"""
        # set vectors what would span the grid for weight-matrices (masks)
        v1 = np.linspace(1,0,dx1).reshape(dx1,1)
        v2 = np.linspace(1,0,dx2).reshape(dx2,1)

        # get raw weight-matrices (masks)
        w_ul = v1@v2.T
        w_ur = np.fliplr(w_ul)
        w_dl = np.flipud(w_ul)
        w_dr = np.fliplr(w_dl)

        # normalize weights so that they sum up to one
        w_tot = w_ul + w_ur + w_dl + w_dr
        w_ul,w_ur,w_dl,w_dr = w_ul/w_tot, w_ur/w_tot, w_dl/w_tot, w_dr/w_tot
        return w_ul,w_ur,w_dl,w_dr

    def get_gradients(self,n1,n2,n3):
        """returns random gradients (n3-D vectors) 
        for every point on n1 x n2 grid"""
        grad = 2*np.random.random((n1,n2,n3)) - 1
        grad = grad/np.repeat(np.sqrt(np.sum(grad**2,axis=2)).reshape(n1,n2,1),
                              2,axis=2) # now it's a unit vector 
        return grad

    def get_perlin(self):
        """returns (n_ver-... x n_hor) mtx: random smooth surface
        (n_ver-... is due to removal of an artificial overlap [see below])"""
        octaves = self.get_octaves()
        grid = self.get_grid(raw=True)
        
        y = np.zeros((self.n_ver,self.n_hor))
        
        for ioc, octave in enumerate(octaves):

            # initialize gradients (randomly oriented unit vector)
            n1,n2,n3 = octave.shape
            grad = self.get_gradients(n1,n2,n3)

            # if circular: make top and bottom rows of cells identical,
            # so that it's possible to smoothly *glue* top and bottom part
            #(that's an artificial overlap, will be removed later)
            if self.circular==True: 
                grad[-2**ioc-1:,:,:] = grad[:2**ioc+1,:,:] 
                grad[:,-2**ioc-1:,:] = grad[:,:2**ioc+1,:]

            # grid step along x1 and x2 for current octave
            dx1 = int((len(grid[:,0,0])-1)/(n1-1))
            dx2 = int((len(grid[0,:,0])-1)/(n2-1))

            # masks for the cell edges in the current octave                      
            # (masks are weight matrices 
            #  that assign '1' to one of the corners in the current cell,
            #  and '0' to other three corners;
            #  values of all other mtx elements are smoothly in-between)
            w_ul,w_ur,w_dl,w_dr = self.get_masks(dx1,dx2)

            # loop over all the cells on the grid and smoothen the edges
            for ix1 in range(n1-1):
                for ix2 in range(n2-1): 

                    # indices for points in the current cell
                    iup    = ix1*dx1
                    ileft  = ix2*dx2
                    idown  = (ix1+1)*dx1
                    iright = (ix2+1)*dx2

                    x = grid[iup:idown,ileft:iright,:]

                    # corners of the current cell in the current octave 
                    x_ul = octave[ix1  ,ix2  ,:]
                    x_ur = octave[ix1  ,ix2+1,:]
                    x_dl = octave[ix1+1,ix2  ,:]
                    x_dr = octave[ix1+1,ix2+1,:] 

                    # find local gradients at the corners of the current cell
                    grad_ul = grad[ix1  ,ix2  ].reshape(1,1,2)
                    grad_ur = grad[ix1  ,ix2+1].reshape(1,1,2)
                    grad_dl = grad[ix1+1,ix2  ].reshape(1,1,2)
                    grad_dr = grad[ix1+1,ix2+1].reshape(1,1,2)

                    # find mask-weighted surface y
                    y[iup:idown,ileft:iright] = \
                        y[iup:idown,ileft:iright] + 2**(-ioc) * (\
                        np.sum(grad_ul*(x-x_ul),axis=2)*self.smoothen(w_ul) + \
                        np.sum(grad_ur*(x-x_ur),axis=2)*self.smoothen(w_ur) + \
                        np.sum(grad_dl*(x-x_dl),axis=2)*self.smoothen(w_dl) + \
                        np.sum(grad_dr*(x-x_dr),axis=2)*self.smoothen(w_dr) )

        # remove the artificial overlap to get "circular" surface 
        y_circ = \
            y[:(self.ver_grid-1)*self.points_at_last_octave**self.num_octaves,
              :(self.hor_grid-1)*self.points_at_last_octave**self.num_octaves]
        return y_circ

    def fig(self, waviness=True, freq_t=0.05, freq_x=0.25, cycles=3, size=(9,9)):
        y_circ = self.get_perlin()
        grid = self.get_grid()
        hor_mesh = grid[:,:,0]
        ver_mesh = grid[:,:,1]

        for frame in range(cycles*y_circ.shape[0]):
            # add morphing waviness 
            #(without it waves would look *solid*...)
            if waviness == True:
                harmonics = 0.2*\
                    np.sin(2*np.pi*freq_x*hor_mesh + 2*np.pi*freq_t*frame)*\
                    np.sin(2*np.pi*freq_x*ver_mesh + 2*np.pi*freq_t*frame)
            else: harmonics = np.zeros(y_circ.shape)                    

            fig = plt.figure(1, size)
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter3D(hor_mesh, ver_mesh, \
                         np.roll(y_circ,frame,axis=1) + harmonics)    
            ax.set_zlim(bottom=-1, top=5)
            plt.pause(0.05)
            if frame != cycles*y_circ.shape[0]-1: plt.clf()

if __name__ == '__main__':
    from mpl_toolkits.mplot3d import Axes3D
    
    pf = PerlinFlow(ver_grid=3, hor_grid=5, 
                    num_octaves=2, points_at_last_octave=4, circular=True)
    #pf.fig(size=(9,7))
    
    # perlin surface (rectangular and radiating)
    surface  = pf.get_perlin() # rollable perlin surface
    surface_rad = np.c_[surface, surface[:,0]]
    surface_rad = np.c_[surface_rad.T, surface_rad[0,:].T].T

    # perlin grid (rectangular and raditing)
    hor,ver = surface.shape[1], surface.shape[0]
    grid     = pf.get_grid() # mesh grid for plotting stuff
    hor_mesh = grid[:,:,0]
    ver_mesh = grid[:,:,1]

    thetas = np.ones((ver+1,1))@\
             np.linspace(0,2*np.pi, hor+1)[np.newaxis,:]
    hor_rad = np.array([np.cos(angle)*np.log(1+1.1**i) 
                        for i in range(ver+1) 
                        for angle in np.linspace(0,2*np.pi, hor+1)]).\
              reshape(ver+1, hor+1)
    ver_rad = np.array([np.sin(angle)*np.log(1+1.1**i) 
                        for i in range(ver+1) 
                        for angle in np.linspace(0,2*np.pi, hor+1)]).\
              reshape(ver+1, hor+1)

    # figure specifications
    contour = False # if True: plots contour-plot, otherwise: plots 3d scatter plot
    radiate = False # if True: noise radiates from the center    

    # parameters for grid and customized waviness on top of the perlin surface
    if radiate: x,y,surface = hor_rad, ver_rad, surface_rad
    else: x,y = hor_mesh, ver_mesh
    freq_x = 0.35 # some parameters for customized_waviness ...
    freq_t = 0.02 #...(the ones here are only applicable to sin-waves)
    
    frames = 150 # number of frames in the animation
    for frame in range(frames):        
        # can add any 2d function here to break the repetitiveness 
        # of the rolling perlin pattern (here I use the same as in the default version)
        #(calculating function values on hor_mesh or ver_mesh 
        # will automatically make it 2d)
        customized_waviness = 0.2*\
            np.cos(2*np.pi*freq_x*x + 2*np.pi*freq_t*frame)*\
            np.cos(2*np.pi*freq_x*y + 2*np.pi*freq_t*frame)
        surface_frame = np.roll(surface,frame,axis=1) + customized_waviness
           
        fig = plt.figure(1, (9,6))
        if contour: 
            plt.contourf(x, y, surface_frame, cmap='bone')
        else:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter3D(x, y, surface_frame) 
            ax.set_zlim(bottom=-1, top=5)       
        plt.pause(0.01)
        if frame != frames-1: plt.clf()