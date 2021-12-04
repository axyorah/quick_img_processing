from typing import List, Tuple, Dict, Set, Optional, Union
import os
import cv2 as cv
import numpy as np


if __name__ == "__main__":
    from perlin_flow import PerlinFlow
    from pattern_utils import (
        Pattern, 
        Polygon, 
        Shape,
        Poly,
        PerlinShape, 
        PerlinComplexShape,
        FrameSqeuence
    )
else:
    from utils.perlin_flow import PerlinFlow
    from utils.pattern_utils import (
        Pattern, 
        Polygon, 
        Shape,
        Poly,
        PerlinShape, 
        PerlinComplexShape,
        FrameSqeuence
    )


class HaSEffect(PerlinComplexShape):
    PATH_VERTICES = os.path.join('utils', 'custom_pattern.txt')
    def __init__(self):
        super().__init__()
        self.perlin_row_idx = 0
        
        self._scale0 = 0.75
        self._star_scale0 = 0.3

        self.children = [self._get_has()] + self._get_stars()

    def _get_has(self):
        with open(self.PATH_VERTICES, 'r') as f:
            lines = f.read().splitlines()

        xy = np.array([
            list(map(int, line.split(' ')))
            for line in lines
        ])
        
        xmax,xmin,ymax,ymin = \
            xy[:,0].max(), xy[:,0].min(), xy[:,1].max(), xy[:,1].min()        
        scale = int(np.mean([xmax - xmin, ymax - ymin]))
        center = np.array([[360, 376]])
            
        xy -= center # center around 0
        xy = xy / scale # fit into a unit circle

        has = PerlinShape(xy, perlin=self.perlin)
        has.perlin_row_idx = self.perlin_row_idx + self.perlin.shape[0] // 2
        
        return has

    def _get_stars(self):
        n_stars, n_poly = 10, 5  
        angle = 2 * np.pi / n_poly * 2
        
        stars = [
            Poly(n_poly, angle=angle) 
            for _ in range(n_stars)
        ]
        
        for i,star in enumerate(stars):
            star.scale(self._star_scale0)
            star.translate([1, 0])
            star = Shape(star.vertices)
            star.rotate(2 * np.pi / n_stars * i)
            
            stars[i] = PerlinShape(
                star.vertices, perlin=self.perlin
            )
            stars[i].perlin_row_idx = self.perlin_row_idx
            
        self._stars = stars
        return self._stars


class SpellEffect(PerlinComplexShape):
    def __init__(self):
        super().__init__()
        self.perlin_row_idx = np.random.choice(self.perlin.shape[0])
        self._scale0 = 0.5

        self.n_tris = 3
        self.n_sq = 6
        self.n_hex = 2
        self.n_circs = 1
        self.n_mtris = 6
        self.n_pents = 1
        
        self.children = (
            self._get_tris()  + 
            self._get_squares() +
            self._get_hex() + 
            self._get_circle() +
            self._get_minitris() + 
            self._get_pents()
        )
        self.colors = [
            (
                int(64*i/self.n_tris + 15*(self.n_tris - i)/self.n_tris),
                int(255*i/self.n_tris + 225*(self.n_tris - i)/self.n_tris),
                int(0*i/self.n_tris + 192*(self.n_tris - i)/self.n_tris)
            ) for i in range(self.n_tris)
        ] + [
            (
                int(29*i/self.n_sq + 245*(self.n_sq - i)/self.n_sq),
                int(230*i/self.n_sq + 157*(self.n_sq - i)/self.n_sq),
                int(181*i/self.n_sq + 10*(self.n_sq - i)/self.n_sq)
            ) for i in range(self.n_sq)
        ] + [
            (
                int(225*i/self.n_hex + 227*(self.n_hex - i)/self.n_hex),
                int(225*i/self.n_hex + 138*(self.n_hex - i)/self.n_hex),
                int(9*i/self.n_hex + 10*(self.n_hex - i)/self.n_hex)
            ) for i in range(self.n_hex)
        ]  + \
            [(221,157,9)]*self.n_circs + \
            [(234,217,0)]*self.n_mtris + \
            [(0,242,255)]*self.n_pents

    def _get_tris(self):
        tri_rad = 0.7 # center of the circle that touches triangle vertices

        return [
            PerlinShape(
                Poly(3).scale( tri_rad ).vertices, 
                perlin=self.perlin,
                perlin_row_idx=i * self.perlin.shape[0]//self.n_tris
            ) for i in range(self.n_tris)
        ]
        
    def _get_squares(self):
        sq_rad = 1.1

        return [
            PerlinShape(
                Poly(4)\
                    .scale( (0.93 if i%2 else 1)*sq_rad )\
                    .vertices,
                perlin=self.perlin,
                perlin_row_idx=self.perlin_row_idx
            ).rotate( i//2 * 2*np.pi/(self.n_sq//2) / 4 )
            for i in range(self.n_sq)
        ]
    
    def _get_hex(self):
        hex_rad = 1.40

        return [
            PerlinShape(
                Poly(6)\
                    .scale( (0.93 if i%2 else 1)*hex_rad )\
                    .vertices,
                perlin=self.perlin,
                perlin_row_idx=0
            ) for i in range(self.n_hex)
        ]
    
    def _get_circle(self):
        circ_rad = [1.15, 1.40, 1.45]

        return [
            PerlinShape(
                Poly(18).scale( circ_rad[i] ).vertices,
                perlin=self.perlin,
                perlin_row_idx=0
            ) for i in range(self.n_circs)
        ]
        
    def _get_minitris(self):
        mtri_rad = 0.2
        mtri_dist = 1.55

        return [
            PerlinShape(
                Poly(3)\
                    .scale( mtri_rad )\
                    .translate([mtri_dist, 0])\
                    .vertices,
                perlin=self.perlin,
                perlin_row_idx=0
            ).rotate(i * 2*np.pi/self.n_mtris) 
            for i in range(self.n_mtris)
        ]
    
    def _get_pents(self):        
        pent_rad = 0.3

        return [
            PerlinShape(
                Poly(5, angle=2*np.pi/5 * 2)\
                    .scale( pent_rad )\
                    .vertices,
                perlin=self.perlin
            ) for i in range(self.n_pents)
        ]

    def draw(self, frame):
        for child, color in zip(self.children, self.colors):
            cv.polylines(
                frame, [child.vertices.astype(int)], True, color, 2
            )


class KaboomEffect(FrameSqeuence):
    PATH_FRAMES_DIR = (
        os.path.join("imgs", "kaboom") if os.path.isdir("imgs")
        else os.path.join("..", "imgs", "kaboom")
    )
    def __init__(self):
        super().__init__()


            
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

    spell = SpellPatternEffect()
    pt1, pt2 = (w//2-rad, h//2-rad), (w//2+rad, h//2+rad)
    spell.draw_pattern(img, pt1, pt2)

    cv.imshow("press 'q' to quit", img)
    key = cv.waitKey(5000)
    if key == ord('q'):
        cv.destroyAllWindows()