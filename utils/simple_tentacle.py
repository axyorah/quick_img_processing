#!/usr/bin/env python3
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import copy
import cv2 as cv

def rotate_notrig(vec: np.ndarray, direction: tuple) -> np.ndarray:
    """rotate (2,1) vector or (2,N) mtx (N vectors) 
    by an angle between Ox and vector with direction (x,y)
    :type vec: (2,N)-array
    :type direction: tuple
    """    
    x,y = direction
    d = np.sqrt(x**2 + y**2)
    co = x/d
    si = y/d
    return np.array([
        [co, -si],
        [si,  co]
    ]).dot(vec)

def rotate(vec: np.ndarray, angle: float) -> np.ndarray:
    """rotate (2,1) vector or (2,N) mtx (N vectors) by given angle around origin
    :type vec: (2,N)-array
    :type angle: float
    """
    return np.array([
        [np.cos(angle), np.cos(np.pi / 2 + angle)],
        [np.sin(angle), np.sin(np.pi / 2 + angle)]
    ]).dot(vec)

    
class SimpleTentacleBuilder:
    def __init__(self, tentacle: 'SimpleTentacle'):
        self.tentacle = tentacle
        
    def num_joints(self, _num_joints: int) -> 'SimpleTentacleBuilder':
        self.tentacle._num_joints = _num_joints
        return self
        
    def segment_decay(self, _segment_decay: float) -> 'SimpleTentacleBuilder':
        assert _segment_decay > 0 and _segment_decay <= 1,\
               "segment_decay should be between 0 and 1"
        self.tentacle._segment_decay = _segment_decay
        return self

    def scale(self, _scale: float) -> 'SimpleTentacleBuilder':
        self.tentacle._scale = _scale
        return self

    def root(self, _root: Union[List[float], np.ndarray]) -> 'SimpleTentacleBuilder':
        """[x,y] coord of the anchor point"""
        self.tentacle._root = _root
        return self

    def arm_angle(self, _arm_angle: Union[float, Tuple[float,float]]) -> 'SimpleTentacleBuilder':
        """angle (float, [radians]) or direction ([x,y] vector) at root"""
        self.tentacle._arm_angle = _arm_angle
        return self

    def max_angle_between_segments(self, _max_angle_between_segments: float) -> 'SimpleTentacleBuilder':
        """in radians"""
        self.tentacle._max_angle_between_segments = _max_angle_between_segments
        return self

    def angle_freq(self, _angle_freq: float) -> 'SimpleTentacleBuilder':
        self.tentacle._angle_freq = _angle_freq
        return self

    def angle_phase_shift(self, _angle_phase_shift: float) -> 'SimpleTentacleBuilder':
        """in radians"""
        self.tentacle._angle_phase_shift = _angle_phase_shift
        return self

    def flip(self, _flip: bool) -> 'SimpleTentacleBuilder':
        self.tentacle._flip = _flip
        return self
    
    def build(self) -> 'SimpleTentacle':
        coeff = 1 - self.tentacle._segment_decay        
        self.tentacle.segments = [
            coeff**i for i in range(self.tentacle._num_joints)
        ]
        return self.tentacle
    

class SimpleTentacle:
    """
    use:
    ```
    t = SimpleTentacle()\
        .set\
            .num_joints(num_joints)\
            .segment_decay(segment_decay)\
            .scale(scale)\
            .root(root)\
            .arm_angle(arm_angle)\
            .max_angle_between_segments(max_angle)\
            .angle_freq(angle_freq)\
            .angle_phase_shift(angle_phase_shift)\
        .build()

    coordinates = t.solve()
    ```
    """
    def __init__(self):        
        self.segments = None
        self._num_joints = 9
        self._segment_decay = 0.1
        self._scale = 1
        self._root = [0,0]
        self._arm_angle = 0
        self._max_angle_between_segments = np.pi/3
        self._angle_freq = 1
        self._angle_phase_shift = 0
        self._flip = False
        
    @property
    def set(self) -> 'SimpleTentacleBuilder':
        return SimpleTentacleBuilder(self)

    def copy(self) -> 'SimpleTentacle':
        return copy.deepcopy(self)
    
    def _get_angles_between_segments(self) -> np.ndarray:
        # tanh modifier is a fugly fix to ensure that
        # no matter the `angle_phase_shift` first segment stays horizontal
        return \
            self._max_angle_between_segments * \
            np.sin(
                self._angle_freq * np.linspace(0, 2*np.pi, self._num_joints) +
                self._angle_phase_shift
            ) * np.tanh(np.linspace(0, 10, self._num_joints) + 1)
            
    def _get_raw_joints(self, angles_between_segments) -> np.ndarray:
        """
        returns coordinates of raw joints connecting segments;
        raw = default scale of 1, default root of (0,0), 
        default arm_angle of 0,
        however all angles between segments are correctly applied
        """
        joints = [np.array([[0,0]]).T] * self._num_joints
        prev_angle = 0
        for i in range(self._num_joints - 1):
            angle = prev_angle + angles_between_segments[i]
            joint = rotate(
                np.array([[self.segments[i],0]]).T, 
                angle
            ) + joints[i]
            joints[i+1] = joint
            prev_angle = angle
        
        return np.concatenate(joints, axis=1)
        
   
    def solve(self) -> np.ndarray:
        """
        returns coordinates of joints (2,num_joints) of the wiggled tentacle
        starting from `root` and 
        rotated by `arm_angle` compared to the flat counterpart.
        Wiggliness is achieved by adjusting the angle between segments:
        the angle follows sin law and can be adjusted via:
        - max_angle_between_segments - ...
        - angle_freq - num of tentacle convex parts
        - angle_phase_shift - ensures that the tentacle flexes both ways
        """
        # get teh angles between segments
        angles_between_segments = self._get_angles_between_segments()
        
        # get raw coordinates to joints between segments
        #(raw: default scale, default flip, default rotation)
        joints = self._get_raw_joints(angles_between_segments)
        
        # flip the tentacle around horizontal axis if needed:
        if self._flip:
            joints = joints * np.array([[1,-1]]).T
        
        # rotate the entire tentacle by `arm_angle`
        #(`arm_angle` = 0: horizontal pointing left->right)
        # and make sure that it's `scale`d starts at `root`
        if isinstance(self._arm_angle, tuple) or isinstance(self._arm_angle, list):
            joints = self._scale * rotate_notrig(
                joints, 
                self._arm_angle
            ) + np.array(self._root).reshape(2,1)
        else:
            joints = self._scale * rotate(
                joints, 
                self._arm_angle
            ) + np.array(self._root).reshape(2,1)

        return joints


class TentaclePainterBuilder:
    def __init__(self, painter: 'TentaclePainter'):
        self.painter = painter

    def tentacle(self, _tentacle: 'SimpleTentacle') -> 'TentaclePainterBuilder':
        self.painter._tentacle = _tentacle
        return self

    def perlin(self, _perlin: np.ndarray) -> 'TentaclePainterBuilder':
        self.painter._perlin = _perlin
        return self

    def anchor(self, _anchor: Union[List[int],Tuple[int,int]]) -> 'TentaclePainterBuilder':
        self.painter._anchor = _anchor
        return self

    def direction(self, _direction: Union[List[int],Tuple[int,int]]) -> 'TentaclePainterBuilder':
        self.painter._direction = _direction
        return self

    def squiggle_coeff(self, _squiggle_coeff: float) -> 'TentaclePainterBuilder':
        self.painter._squiggle_coeff = _squiggle_coeff
        return self
    
    def scale(self, _scale: Union[int, float]) -> 'TentaclePainterBuilder':
        self.painter._scale = _scale
        return self

    def color(self, _color: Tuple[int,int,int]) -> 'TentaclePainterBuilder':
        self.painter._color = _color
        return self

    def build(self) -> 'TentaclePainter':
        return self.painter

class TentaclePainter():
    """
    use:
    ```
    painter = TentaclePainter()\
        .set\
            .tentacle(tentacle)\ # SimpleTentacle instance
            .perlin(perlin)\ # perlin matrix [np.ndarray]
            .anchor(anchor)\ # (x,y) coordinates of anchor (root) point
            .direction(direction)\ # (x,y) coordinates of direction vec
            .squiggle_coeff(scoeff)\ # num between 0 and 1, determines "squiggliness"
            .scale(scale)\ # scale coeff [int or float]
            .color(color)\ # (b,g,r) 
        .build()

    painter.paint(frame) # pass image array (height, width, num_channels)
    painter.increment_counter() # needed to choose `next` element in perlin mtx
    ```
    """
    def __init__(self):
        self._tentacle = None
        self._perlin = None
        self._anchor = None
        self._direction = None
        self._squiggle_coeff = None
        self._scale = None
        self._color = None
        self.counter = 0

    @property
    def set(self) -> 'TentaclePainterBuilder':
        return TentaclePainterBuilder(self)
    
    def increment_counter(self) -> int:
        self.counter += 1
        return self.counter

    def paint(self, frame: np.array) -> None:
        perlin_random = self._perlin[
            self._tentacle._perlin_idx, 
            self.counter % self._perlin.shape[1]
        ]
        scale_ref = self._tentacle._scale # cache scale!
        self._tentacle\
            .set\
                .scale(int(np.round(self._scale * self._tentacle._scale)))\
                .root(self._anchor)\
                .arm_angle(self._direction)\
                .max_angle_between_segments(self._squiggle_coeff * np.pi * perlin_random)\
                .angle_freq(1 * perlin_random)\
                .angle_phase_shift(2 * np.pi * perlin_random)\
            .build()
            
        coords = self._tentacle.solve().astype(int)
    
        for i in range(coords.shape[1]-1):
            cv.line(
                frame, 
                tuple(coords[:,i]), 
                tuple(coords[:,i+1]), 
                self._color or self._tentacle._color, 
                int(np.round(self._scale * self._tentacle._thickness[i]))
            )

        # reset scale!!!
        self._tentacle.set.scale(scale_ref)


def main():
    import matplotlib.pyplot as plt
    from perlin_flow import PerlinFlow

    num_tentacles = 8
    
    roots = np.array([
            np.cos(np.linspace(0,2*np.pi,num_tentacles+1)),
            np.sin(np.linspace(0,2*np.pi,num_tentacles+1))
    ]).T[:-1]
            
    arm_angles = [
            2*np.pi/num_tentacles * it
            for it in range(num_tentacles)
    ]
            
    tentacles = [
        SimpleTentacle()\
        .set\
            .num_joints(15)\
            .segment_decay(0.1)\
            .scale(1)\
            .root(roots[it])\
            .arm_angle( arm_angles[it] )\
            .max_angle_between_segments(np.pi/3)
            .angle_freq(1)
            .angle_phase_shift(0)
        .build() for it in range(num_tentacles)
    ]
    
    pf = PerlinFlow(ver_grid=4, hor_grid=4, points_at_last_octave=6)
    perlin = pf.get_perlin()
    perlin = (perlin - perlin.min()) / (perlin.max() - perlin.min())
    
    perlin_idx = np.random.choice(range(perlin.shape[0]), num_tentacles)
    
    for iframe in range(400):
        plt.clf()
        for it in range(num_tentacles):

            perlin_random = perlin[
                perlin_idx[it],
                iframe % perlin.shape[1]
            ]
            
            coords = tentacles[it]\
                .set\
                    .max_angle_between_segments( np.pi/3*perlin_random )\
                    .angle_freq( 1* perlin_random )\
                    .angle_phase_shift( 2*np.pi*perlin_random )\
                .build()\
                .solve()
                
            plt.plot(coords[0,:], coords[1,:], 'o-', c=(0,0,0))
            
        plt.xlim([-7,7])
        plt.ylim([-7,7])
        plt.pause(0.001)


if __name__ == "__main__":
    main()