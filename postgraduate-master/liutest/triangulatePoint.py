'''
import cv2
from numpy import *
P1 = eye(4)
P2 = array([[ 0.878, -0.01 ,  0.479, -1.995],
            [ 0.01 ,  1.   ,  0.002, -0.226],
            [-0.479,  0.002,  0.878,  0.615],
            [ 1.   ,  1.   ,  1.   ,  1.   ]])
# Homogeneous arrays
a3xN = array([[ 0.091,  0.167,  0.231,  0.083,  0.154],
              [ 0.364,  0.333,  0.308,  0.333,  0.308],
              [ 1.   ,  1.   ,  1.   ,  1.   ,  1.   ]])
b3xN = array([[ 0.42 ,  0.537,  0.645,  0.431,  0.538],
              [ 0.389,  0.375,  0.362,  0.357,  0.345],
              [ 1.   ,  1.   ,  1.   ,  1.   ,  1.   ]])
# The cv2 method
X = cv2.triangulatePoints( P1[:3], P2[:3], a3xN[:2], b3xN[:2] )  # coor 
# Remember to divide out the 4th row. Make it homogeneous
X /= X[3]
# Recover the origin arrays from PX
x1 = dot(P1[:3],X)
x2 = dot(P2[:3],X)
# Again, put in homogeneous form before using them
x1 /= x1[2]
x2 /= x2[2]
 
print('X\n',X)
print('x1\n',x1)
print('x2\n',x2)
'''

import cv2
import numpy as np

cameraMatrix = np.array([[653.53,0,405],[0,623.26,267.43],[0,0,1]])
obj_points = np.array([[0.0,0.0,0.0],[8.0,0.0,0.0],[0.0,6.0,0.0],[8.0,6.0,0.0]]) # 4*3
print(type(obj_points),obj_points.shape,obj_points)
img_points = np.array([[[100.0,100.0]],[[100.0,200.0]],[[200.0,100.0]],[[200.0,200.0]]]) #4 * 1 * 2
print(type(img_points),img_points.shape,img_points)

ret, matrix, distortion, rotation, translation = \
    cv2.calibrateCamera([obj_points], [img_points], gray.shape[::-1], cameraMatrix,None)