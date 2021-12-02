# -*- coding: utf-8 -*-
from typing import List, Tuple, Dict, Set, Optional, Union
import os
import cv2 as cv
import numpy as np
try:
    from utils.perlin_flow import PerlinFlow
except:
    from perlin_flow import PerlinFlow


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

class HaSPatternEffect:
    def __init__(self):        
        self.center = np.array([[0,0]])
        self.perlin = PerlinFlow().get_perlin()
        self.perlin_idx = np.random.choice(self.perlin.shape[0])
        
        self.scale0 = 0.75
        self.n_stars = 10
        self.rad_stars = 0.3
        self.star_dist_from_center = 1.
        
        self.redpattern = self.get_red()        
        self.starpatterns = self.get_stars(
                n_stars=self.n_stars, 
                rad_stars = self.rad_stars*self.scale0,                
                dist_from_center=self.star_dist_from_center*self.scale0)  
        
        self.patterns = [self.redpattern] + self.starpatterns
        
    def get_red(self):
        RED_PATH = "./utils/custom_pattern.txt" if os.path.isfile("./utils/custom_pattern.txt") \
            else "./custom_pattern.txt"
        with open(RED_PATH, "r") as f:
            xy = []
            lines = f.read().splitlines()
            for line in lines:
                x,y = map(int, line.split(" "))
                xy.append([x,y])
            xy = np.array(xy)
            xmax,xmin,ymax,ymin = \
                xy[:,0].max(), xy[:,0].min(), xy[:,1].max(), xy[:,1].min()
            #center = np.array([[(xmax+xmin)//2, (ymax+ymin)//2]])
            center = np.array([[360, 376]])
            scale = int(np.mean([xmax - xmin, ymax - ymin]))
            
            xy -= center # center around 0
            xy = xy / scale # fit into a unit circle
            xy *= self.scale0 # fit into a circle with needed rad
            
        return Pattern(xy, center)  
        
    def get_stars(self, n_stars=10, rad_stars=0.3, dist_from_center=1.3):
        stars = [Polygon(5, self.center, 
                         rad_stars, 
                         dist_from_center=dist_from_center,
                         perlin=self.perlin, 
                         perlin_idx=self.perlin_idx)
                 for i in range(n_stars)]
        for i,star in enumerate(stars):        
            star.initialize_vertices(angles=[2*i * 2*np.pi/5 for i in range(6)])
            star.initialize_dists()
            star.rotate_vertices(i * 2*np.pi/n_stars)
        return stars
        
    def draw_pattern(self, frame, pt1, pt2, fill=True):
        x1,y1 = pt1
        x2,y2 = pt2
        
        center = np.array([[(x2+x1)//2, (y2+y1)//2]])
        scale = max((x2-x1),(y2-y1))
            
        for i,pattern in enumerate(self.patterns):
            vertices = pattern.update_vertices(center=center, scale=scale)
            if fill:
                cv.fillPoly(frame, [vertices], (0,0,255))
            else:
                cv.polylines(frame, [vertices], True, (0,0,255), 2)
                
                
class SpellPatternEffect:
    def __init__(self):
        self.scale0 = 0.5
        self.center = np.array([[0,0]])
        self.perlin = PerlinFlow().get_perlin()
        self.perlin_idx = np.random.choice(self.perlin.shape[0])
        
        self.n_tris = 3
        self.n_sq = 6
        self.n_hex = 2
        self.n_circs = 1
        self.n_mtris = 6
        self.n_pents = 1
        
        self.tris = self.get_tris()
        self.squares = self.get_squares()
        self.hexes = self.get_hex()
        self.circles = self.get_circle()
        self.mtris = self.get_minitris()
        self.pents = self.get_pents()
        
        self.patterns = \
            self.tris  + self.squares + \
            self.hexes + self.circles + \
            self.mtris + self.pents
            
        self.colors = \
            [
                (
                    int(64*i/self.n_tris + 15*(self.n_tris-i)/self.n_tris),
                    int(255*i/self.n_tris + 225*(self.n_tris-i)/self.n_tris),
                    int(0*i/self.n_tris + 192*(self.n_tris-i)/self.n_tris)
                ) for i in range(self.n_tris)
            ] + \
            [
                (
                    int(29*i/self.n_sq + 245*(self.n_sq-i)/self.n_sq),
                    int(230*i/self.n_sq + 157*(self.n_sq-i)/self.n_sq),
                    int(181*i/self.n_sq + 10*(self.n_sq-i)/self.n_sq)
                ) for i in range(self.n_sq)
            ] + \
            [
                (
                    int(225*i/self.n_hex + 227*(self.n_hex-i)/self.n_hex),
                    int(225*i/self.n_hex + 138*(self.n_hex-i)/self.n_hex),
                    int(9*i/self.n_hex + 10*(self.n_hex-i)/self.n_hex)
                ) for i in range(self.n_hex)
            ]   + \
                [(221,157,9)]*self.n_circs + \
                [(234,217,0)]*self.n_mtris + \
                [(0,242,255)]*self.n_pents
        
        
    def get_tris(self):
        tri_rad = 0.7 * self.scale0; # center of the circle that touches triangle vetices
        
        perlin_ids = [
            i * self.perlin.shape[0]//self.n_tris 
            for i in range(self.n_tris)
        ]

        return [
            Polygon(
                3, self.center, tri_rad, 
                perlin=self.perlin, perlin_idx=perlin_ids[i]
            ) for i in range(self.n_tris)
        ]
        
    def get_squares(self):
        sq_rad = 1.1 * self.scale0
        perlin_ids = [np.random.choice(self.perlin.shape[0])] * self.n_sq
        
        squares = [
            Polygon(
                4, self.center, (0.93 if i%2 else 1)*sq_rad, 
                perlin=self.perlin, perlin_idx=perlin_ids[i]
            ) for i in range(self.n_sq)
        ]

        for i in range(0,self.n_sq,2):
            squares[i  ].rotate_vertices(i//2 * 2*np.pi/(self.n_sq//2) / 4)
            squares[i+1].rotate_vertices(i//2 * 2*np.pi/(self.n_sq//2) / 4)
        
        return squares
    
    def get_hex(self):
        hex_rad = 1.40 * self.scale0
        
        perlin_ids = [
            i * self.perlin.shape[0] // (self.n_hex//2) 
            for i in range(self.n_hex//2) for _ in range(2)
        ]

        return [
            Polygon(
                6, self.center, (0.93 if i%2 else 1)*hex_rad, 
                perlin=self.perlin, perlin_idx=perlin_ids[i]
            ) for i in range(self.n_hex)
        ]
    
    def get_circle(self):
        circ_rad = [i * self.scale0 for i in [1.15, 1.40, 1.45]]

        return [
            Polygon(18, self.center, circ_rad[i], perlin=self.perlin) 
            for i in range(self.n_circs)
        ]
        
    def get_minitris(self):
        mtri_rad = 0.2*self.scale0
        mtri_dist = 1.55 * self.scale0
        
        mtris = [
            Polygon(
                3, self.center, mtri_rad, 
                dist_from_center=mtri_dist, perlin=self.perlin
            ) for i in range(self.n_mtris)
        ]

        for i,mtri in enumerate(mtris):
            mtri.rotate_vertices(i * 2*np.pi/self.n_mtris)

        return mtris
    
    def get_pents(self):        
        pent_rad = 0.3*self.scale0

        pents = [
            Polygon(
                5, self.center, pent_rad, perlin=self.perlin, 
                perlin_idx=np.random.choice(self.perlin.shape[0])
            ) for i in range(self.n_pents)
        ] 
        
        for pent in pents:
            pent.initialize_vertices(
                angles=[2*i * 2*np.pi/5 for i in range(6)]
            )
            pent.initialize_dists()

        return pents
    
    def draw_pattern(self, frame, pt1, pt2):
        x1,y1 = pt1
        x2,y2 = pt2
        
        center = np.array([[(x2+x1)//2, (y2+y1)//2]])
        scale = max((x2-x1),(y2-y1))
        
        for i,pattern in enumerate(self.patterns):
            vertices = pattern.update_vertices(
                center=center, 
                scale=scale
            )
            cv.polylines(
                frame, [vertices], True, self.colors[i], 2
            )
            
class KaboomPatternEffect:
    def __init__(self):
        self.fade = 5 # num of kaboom frames that gradually fade at the end
        self.load_frames()
        self.isongoing = False
        self.ongoingframe = 0
        self.duration = len(self.kabooms)
        
        # only show anymation if jutsu was detection in at least 5/10 last frames
        self.detectionque = [False]*10
        self.detectionthreshold = 5
        
    def load_frames(self):
        KABOOM_DIR = os.path.join("imgs", "kaboom") if os.path.isdir("imgs") \
            else os.path.join("..", "imgs", "kaboom")
        
        combined = [
            cv.imread(
                os.path.join(KABOOM_DIR, f"transp_kaboom{i}.png"),
                cv.IMREAD_UNCHANGED
            ) for i in range(1,20)
        ]

        self.kabooms = [comb[:,:,:3] for comb in combined]
        self.masks   = [comb[:,:,3] for comb in combined]        
        
    def draw_pattern(self, frame, detected, pt1=(None,None), pt2=(None,None)):
        self.detectionque.pop(0)
        self.detectionque.append(detected)
            
        if sum(self.detectionque) >= self.detectionthreshold:
            self.isongoing = True
            
        if self.isongoing:
            if self.ongoingframe < self.duration:
                # get get correct kaboom img and corresponding mask
                kaboom = self.kabooms[self.ongoingframe]
                mask = self.masks[self.ongoingframe]                
                
                # overlay kaboom with current frame with correct mask and slow kaboom fade at the end
                # without the fade we could've also use:
                # cv.copyTo(kaboom, mask, frame)
                # but to do the correct fading we need to use the intermediate `combined`:
                fadedegree = max(
                    0, 
                    self.fade - (self.duration - self.ongoingframe) + 1
                ) / self.fade # fade of kaboom

                combined = frame.copy()
                cv.copyTo(kaboom, mask, combined)
                cv.addWeighted(
                    combined, 1 - fadedegree, frame, fadedegree, 0, frame
                )

                # increment the kaboom animation frame count
                self.ongoingframe += 1
            else:
                self.isongoing = False
                self.ongoingframe = 0

class LightningPatternEffect:
    def __init__(self):
        self.fg = self.get_frame_generator()
        self.isongoing = False
        self.ongoingframe = 0
        
        # only show animation if jutsu was detection in at least 5/10 last frames
        self.detectionque = [False]*10
        self.detectionthreshold = 5
        self.modduration = 4 # num of lightning frames per animation

        self.pt = (None,None)

    def get_frame_generator(self):
        LIGHTNING_DIR = os.path.join("imgs","lightning") if os.path.isdir("imgs") \
            else os.path.join("..", "imgs", "lightning")

        num_bundles = 8

        # we'll use only 4 out of original 10 frames
        frame_indices = [1,4,7,9]        
        
        # preload all frames
        frame_bundles = [[] for _ in range(num_bundles)]
        for i in range(num_bundles):
            for j in frame_indices:
                fullimname = os.path.join(
                    LIGHTNING_DIR, f"lightning{i}-{j}.png"
                )
                fullmaskname = os.path.join(
                    LIGHTNING_DIR, f"mask{i}-{j}.png"
                )
            
                # read
                frame = cv.imread(fullimname)
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mask = cv.imread(fullmaskname)

                # upscale x2
                frame = cv.resize(frame, (640*2, 480*2))
                mask = cv.resize(mask, (640*2, 480*2))
            
                # save
                frame_bundles[i].append((frame, mask))

            # add empty frame/mask at a random loc to simulate flickering
            empty = np.zeros(frame.shape, dtype=np.uint8)
            idx = np.random.randint(0,len(frame_bundles[i]))
            frame_bundles[i].insert(idx, (empty, empty))
            frame_bundles[i].extend([(empty, empty), (empty, empty)])
            
        # yield all frames from randomly chosen bundle 
        # maybe flip the frames 
        while True:
            i = np.random.randint(0,num_bundles)
            flipud = np.random.randint(0,2)
            fliplr = np.random.randint(0,2)
            for frame, mask in frame_bundles[i]:            
                if flipud:
                    frame = np.flipud(frame)
                    mask = np.flipud(mask)
                if fliplr:
                    frame = np.fliplr(frame)
                    mask = np.fliplr(mask)
                yield frame, mask
        
    def draw_pattern(self, frame, detected, pt1=(None,None), pt2=(None,None)):
        h,w,c = frame.shape
        if pt1 != (None,None) and pt2 != (None,None):
            self.pt = ((pt1[0] + pt2[0])//2, (pt1[1] + pt2[1])//2)

        self.detectionque.pop(0)
        self.detectionque.append(detected)
            
        self.isongoing = sum(self.detectionque) >= self.detectionthreshold
            
        if self.isongoing or self.ongoingframe % self.modduration:
            # get get correct lightning img and corresponding mask
            lightning, mask = next(self.fg)            
            lightning = cv.cvtColor(lightning, cv.COLOR_BGR2RGB)

            background = frame.astype(float)/255
            foreground = np.zeros(frame.shape)
            alpha = np.zeros(frame.shape)

            # center lightning at avg(pt1,pt2) and add it to foreground                  
            #MR: lightning and mask are twice as big as frame wrt height and width!
            ll,rl = w - self.pt[0], 2*w - self.pt[0]
            ul,dl = h - self.pt[1], 2*h - self.pt[1]

            foreground += lightning[ul:dl,ll:rl,:].astype(float)/255
            alpha += mask[ul:dl,ll:rl,:].astype(float)/255

            # overlay lightning (foregr) with current frame (backgr) with correct mask (alpha)
            foreground = cv.multiply(alpha, foreground)
            background = cv.multiply(1. - alpha, background)

            # modify original frame 
            combined = cv.add(foreground, background)
            cv.copyTo(
                (255*combined).astype(np.uint8), 
                np.ones(combined.shape, dtype=np.uint8), 
                dst=frame)

            # increment the lightning animation frame count
            self.ongoingframe += 1
        else:
            self.ongoingframe = 0


if __name__ == '__main__':
    w,h, rad = 640, 480, 50
    img = 255 * np.ones((h, w, 3), dtype=int)

    DUMMY_PATH = "./imgs/dummy_img.jpg" if os.path.isdir("./imgs") else "../imgs/dummy_img.jpg"
    cv.imwrite(DUMMY_PATH, img)
    img = cv.imread(DUMMY_PATH) # now it's cv::UMat

    hand_pattern = HandPatternEffect()
    pt1, pt2 = (w//2-rad, h//2-rad), (w//2+rad, h//2+rad)
    hand_pattern.draw_pattern(img, pt1, pt2)

    cv.imshow("press 'q' to quit", img)
    key = cv.waitKey(5000)
    if key == ord('q'):
        cv.destroyAllWindows()