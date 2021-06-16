#!/usr/bin/env python3
"""
Created on Fri Aug  2 17:42:09 2019

@author: axeh
"""
import numpy as np

def rotate_notrig(vec: np.ndarray, direction: tuple):
    """rotate (2,1) vector or (2,N) mtx (N vectors) 
    by an angle between Ox and vector with direction (x,y)
    :type vec: (2,N)-array
    :type direction: tuple
    """    
    x,y = direction
    d = np.sqrt(x**2 + y**2)
    co = x/d
    si = y/d
    return np.array([[co, -si],
                     [si,  co]]).dot(vec)

def rotate(vec, angle):
    """rotate (2,1) vector or (2,N) mtx (N vectors) by given angle around origin
    :type vec: (2,N)-array
    :type angle: float
    """
    return np.array([[np.cos(angle), np.cos(np.pi / 2 + angle)],
                     [np.sin(angle), np.sin(np.pi / 2 + angle)]]).dot(vec)

    
class SimpleTentacle:
    def __init__(self, num_joints, segment_decay=0.1):
        assert segment_decay > 0 and segment_decay <= 1,\
               "segment_decay should be between 0 and 1"
        self.num_joints = num_joints
        self.segment_decay = segment_decay
        self.flat_tentacle = self.get_flat_tentacle()
        
        
        
    def get_flat_tentacle(self):
        """len of first segment = 1"""
        coeff = 1 - self.segment_decay
        
        segments = [coeff**i for i in range(self.num_joints)]
        self.segments = segments
        
        # return coordinates of joints of the flat horizontal tentacle
        return np.array([[sum(segments[:i]),0]
                          for i in range(self.num_joints)]).T
    
    def get_wiggly_tentacle(self, scale, root, arm_angle=0,
                            max_angle_between_segments=np.pi/3,
                            angle_freq=1,
                            angle_phase_shift=0,
                            flip=False):
        """
        returns coordinates of joints (2,num_joints) of the wiggled tentacle
        starting from `root` and 
        rotated by `arm_angle` compared to the flat counterpart.
        Wiggliness is achieved by adjusting the angle between segments:
        the angle follows sin law and can be adjusted via:
        - max_angle_between_segments - ...
        - angle_freq - num of tentacle flexes (humps?)
        - angle_phase_shift - ensures that the tentacle flexes both ways
        """
        # get teh angles between segments
        angles_between_segments = \
            max_angle_between_segments * \
            np.sin(angle_freq * np.linspace(0,2*np.pi,self.num_joints) +
                   angle_phase_shift) * \
            np.tanh(np.linspace(0,10,self.num_joints) + 1)
        # tanh modifier is a fugly patch to ensure that
        # no matter the `angle_phase_shift` first segment stays horizontal
        
        # get coordinates to joints of wiggled tentacle
        # assuming that tentacle's starting position is horizontal
        joints = [np.array([[0,0]]).T] * self.num_joints
        prev_angle = 0
        for i in range(self.num_joints-1):
            angle = prev_angle + angles_between_segments[i]
            joint = rotate(np.array([[self.segments[i],0]]).T, angle) +\
                    joints[i]
            joints[i+1] = joint
            prev_angle = angle
        
        joints = np.concatenate(joints, axis=1)
        
        # flip the tentacle around horizontal axis if needed:
        if flip:
            joints = joints * np.array([[1,-1]]).T
        
        # rotate the entire tentacle by `arm_angle`
        #(`arm_angle` = 0: horizontal pointing left->right)
        # and make sure that it's `scale`d starts at `root`
        if isinstance(arm_angle, tuple):
            joints = scale * rotate_notrig(joints, arm_angle) +\
                             np.array(root).reshape(2,1)
        else:
            joints = scale * rotate(joints, arm_angle) +\
                             np.array(root).reshape(2,1)

        return joints


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from perlin_flow import PerlinFlow

    st = SimpleTentacle(17, segment_decay=0.1)
    
    # get a snapshot of a single wiggly tentacle
    wiggly = st.get_wiggly_tentacle(1, (0,0), arm_angle=0,
                            max_angle_between_segments=np.pi/3,
                            angle_freq=1,
                            angle_phase_shift=0)
    plt.clf()
    plt.plot(wiggly[0,:], wiggly[1,:], 'o-', c=(0,0,0))
    plt.pause(5)
    
    # simulate several tentacles
    num_tentacles = 8
    roots = np.array([np.cos(np.linspace(0,2*np.pi,num_tentacles+1)),
                      np.sin(np.linspace(0,2*np.pi,num_tentacles+1))]).T[:-1]
    
    pf = PerlinFlow(ver_grid=4, hor_grid=4, points_at_last_octave=6)
    perlin = pf.get_perlin()
    perlin = (perlin - perlin.min()) / (perlin.max() - perlin.min())
    
    perlin_idx = np.random.choice(range(perlin.shape[0]), num_tentacles)
    
    for iframe in range(400):
        plt.clf()
        for it in range(num_tentacles):

            perlin_random = perlin[perlin_idx[it],
                                   iframe % perlin.shape[1]]
            tentacle = st.get_wiggly_tentacle(
                    1, roots[it],arm_angle=2*np.pi/num_tentacles * it,
                    max_angle_between_segments=np.pi/3*perlin_random,
                    angle_freq=1*perlin_random,
                    angle_phase_shift=2*np.pi*perlin_random
                    )
            plt.plot(tentacle[0,:], tentacle[1,:], 'o-', c=(0,0,0))
        plt.xlim([-7,7])
        plt.ylim([-7,7])
        plt.pause(0.001)