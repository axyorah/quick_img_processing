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
        np.array([
            [np.cos(angle), np.cos(np.pi/2 + angle)],
            [np.sin(angle), np.sin(np.pi/2 + angle)]
        ]),        
        vecs.T
    ).T

class Shape:
    """
    simple geometrical shepe given by a list of vertices 
    as (x,y) coordinates or (N,2)-ndarary;
    vertices are assumed to be given in **relative** coordinates - 
    relative to the center of rotation;
    shape can be translated, uniformely scale and rotated;
    do note that translation is not appied directly
    but calculated on the fly when you request vertices;
    because of that if want your shape to rotate around
    different center you need to use slightely convoluted syntax:

    Suppose, you have the following square shape with side = 2:
    ```
    >>> sq = Shape([
        [ 1, 1],
        [ 1,-1],
        [-1,-1],
        [-1, 1],
        [ 1, 1]
    ])
    ```

    It's centered at (0,0), and if you apply 60 deg rotation
    you get:
    ```
    >>> sq.rotate(np.pi / 3)
    >>> sq.vertices
    array([[-0.3660254,  1.3660254],
           [ 1.3660254,  0.3660254],
           [ 0.3660254, -1.3660254],
           [-1.3660254, -0.3660254],
           [-0.3660254,  1.3660254]])
    ```
    No surprises here.

    Now, suppose, you want to rotate the square around the point 
    external to it, e.g., (-1,0). The correct way of doing it
    is as follows:
    ```
    >>> sq.translate([1,0])     # this moves the shape center
    >>> sq = Shape(sq.vertices) # this creates new shape with external rotation center
    >>> sq.rotate(np.pi / 3)    # new shape is rotated around (0,0)
    >>> sq.translate([-1,0])    # move rotation center back to (-1,0)
    >>> sq.vertices
    array([[-0.8660254 ,  2.23205081],
           [ 0.8660254 ,  1.23205081],
           [-0.1339746 , -0.5       ],
           [-1.8660254 ,  0.5       ],
           [-0.8660254 ,  2.23205081]])
    ```

    Here's a more streamlined way of doing the same:
    ```
    >>> vertices = [
        [ 1, 1],
        [ 1,-1],
        [-1,-1],
        [-1, 1],
        [ 1, 1]
    ]

    >>> sq = Shape(
        Shape(vertices)\
            .translate([ 1, 0 ])\
            .vertices
    )\
        .rotate( np.pi / 3 )\
        .translate( [-1, 0] )

    >>> sq.vertices
    array([[-0.8660254 ,  2.23205081],
           [ 0.8660254 ,  1.23205081],
           [-0.1339746 , -0.5       ],
           [-1.8660254 ,  0.5       ],
           [-0.8660254 ,  2.23205081]])
    ```
    """
    def __init__(self, vertices: Union[np.ndarray,List[List[float]]]):
        self._center = np.array([[0, 0]], dtype=float)
        self._vertices = np.array(vertices).reshape(-1,2)
        self._dist0 = self._get_avg_dist_from_center()
        self._scale0 = 1
        
    def _get_avg_dist_from_center(self) -> float:
        self._dist = np.linalg.norm(self._vertices, axis=1).mean()
        return self._dist        
        
    def translate(self, vec: Union[np.ndarray,List[float]]) -> 'Shape':
        """move shape in a direction specified by `vec` (x,y coord)"""
        self._center += np.array(vec, dtype=float).reshape(-1, 2)
        return self
    
    def scale(self, scalar: float, absolute: bool = True) -> 'Shape':
        """scale by a `scalar` in all dimensions"""
        if absolute:
            coeff = self._dist0 / self._get_avg_dist_from_center()
        else:
            coeff = 1
        self._vertices *= (scalar * coeff)
        return self
    
    def rotate(self, angle: float) -> 'Shape':
        """rotate shape around the angle [radians]"""
        self._vertices = rotate(self._vertices, angle)
        return self
    
    @property
    def vertices(self) -> np.ndarray:
        return self._vertices + self._center


class Poly(Shape):
    def __init__(self, num_vertices: int, angle: float = None):
        self.num_vertices = num_vertices
        self.angle = angle
        self._get_init_vertices()            
        super().__init__( self._vertices )

    def _get_init_vertices(self) -> np.ndarray:
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
    def __init__(
        self, 
        vertices: Union[np.ndarray,List[List[float]]], 
        perlin: Optional[np.ndarray] = None, 
        perlin_modifier: int = 1, 
        perlin_row_idx: Optional[int] = None
    ):
        super().__init__(vertices)
        self.perlin = perlin if perlin is not None else PerlinFlow().get_perlin()
        self.perlin_modifier = perlin_modifier
        self.perlin_col_idx = 0
        self.perlin_row_idx = (
            perlin_row_idx if perlin_row_idx is not None
            else np.random.choice(self.perlin.shape[0])
        ) 
        
    def next(self) -> 'PerlinShape':
        """
        select next column in perlin matrix (perlin_col_idx += 1)
        and rotate by the angle corresponding to
        `perlin_matrix[perlin_row_idx][perlin_col_idx]`
        """
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
    def __init__(self, perlin: Optional[np.ndarray] = None):
        self.perlin = perlin if perlin is not None else PerlinFlow().get_perlin()
        self.children = [] # each child is `PerlinShape` with `next()` method

    def translate(
        self, 
        pt1: Tuple[int,int], 
        pt2: Tuple[int,int]
    ) -> 'PerlinComplexShape':
        """translate into the center of the box between pt1 and pt2"""
        x1,y1 = pt1
        x2,y2 = pt2
        
        center = np.array([[(x2+x1)//2, (y2+y1)//2]])
        
        for child in self.children:
            child._center = center

        return self
    
    def scale(
        self, 
        pt1: Tuple[int,int], 
        pt2: Tuple[int,int]
    ) -> 'PerlinComplexShape':
        """scale to fit max dimension of the boxe given by pt1 and pt2"""
        x1,y1 = pt1
        x2,y2 = pt2
        
        scale = max((x2-x1),(y2-y1))

        for child in self.children:
            child.scale(scale * self._scale0)

        return self
    
    def next(self) -> 'PerlinComplexShape':
        """
        select next column in perlin matrix (perlin_col_idx += 1)
        and rotate by the angle corresponding to
        `perlin_matrix[perlin_row_idx][perlin_col_idx]`
        """
        for child in self.children:
            child.next()

        return self
    
    @property
    def vertices(self) -> List[np.ndarray]:
        return [child.vertices for child in self.children]

    def draw(self, frame: np.ndarray) -> None:
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

    def load(self) -> None:
        """
        load image files from dir specified by `PATH_FRAMES_DIR`
        """
        for name in os.listdir(self.PATH_FRAMES_DIR):
            fname = os.path.join(self.PATH_FRAMES_DIR, name)
            if not os.path.isfile(fname):
                continue

            combined = cv.imread(fname, cv.IMREAD_UNCHANGED)
            self.frames.append(combined[:,:,:3])
            if combined.shape[2] > 3:
                self.masks.append(combined[:,:,3])

    @property
    def isongoing(self) -> bool:
        return self._isongoing

    def maybe_begin(self) -> None:
        """
        set `isongoing` flag to True and reset the frame counter
        IF frame sequence is not already ongoing
        """
        if self.isongoing:
            return
        else:
            self._isongoing = True
            self._ongoingframe = 0

    def maybe_draw(
        self, 
        frame: np.ndarray, 
        pt1: Optional[Tuple[float,float]], 
        pt2: Optional[Tuple[float,float]]
    ) -> None:
        """
        draw current sequence frame on provided image `frame`
        IF there's an ongoing sequence;
        you always want to call `maybe_begin()` 
        right before `maybe_draw(.)`
        """
        if not self.isongoing:
            return
        if self._ongoingframe >= len(self.frames):
            self._isongoing = False
            self._ongoingframe = 0
            return

        effect = self.frames[self._ongoingframe]
        mask = (
            self.masks[self._ongoingframe] if self.masks
            else 255 * np.ones(frame.shape[:2], dtype=int)
        )
        
        fadedegree = max(
            0, 
            self.fade - (len(self.frames) - self._ongoingframe) + 1
        ) / self.fade 

        combined = frame.copy()
        cv.copyTo(effect, mask, combined)
        cv.addWeighted(
            combined, 1 - fadedegree, frame, fadedegree, 0, frame
        )

        self._ongoingframe += 1

