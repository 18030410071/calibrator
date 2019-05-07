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


def getProjMtx(rvec,tvec):
    rmtx, _ = cv2.Rodrigues(rvec)
    return np.hstack((rmtx,tvec))

cameraMatrix = np.array([[653.53, 0.0, 405.88],[0.0, 623.26, 267.43],[0.0, 0.0, 1]],dtype = np.float32)
#print(type(obj_points),obj_points.shape,obj_points)
row = 2
col = 2

obj_points1 = np.zeros((row*col, 3), np.float32)
obj_points1[:,:2] = np.mgrid[0:2:2j, 0:2:2j].T.reshape(-1,2) # 2*2
print(obj_points1)


img_points = np.array([[[100.0, 100.0]],[[140.0, 100.0]],[[100.0, 140.0]],[[140.0, 140.0]]],dtype = np.float32) #4 * 1 * 2
#img_points = np.array([[[100.0, 100.0]],[[140.0, 100.0]],[[180.0, 100.0]],[[100.0, 140.0]],[[140.0, 140.0]],[[180.0, 140.0]],[[100.0, 180.0]],[[140.0, 180.0]],[[180.0, 180.0]]],dtype = np.float32) #4 * 1 * 2
#img_points = np.array(impo) #4 * 1 * 2
#print(type(img_points),img_points.shape,img_points)



ret, matrix, distortion, rotation, translation = \
    cv2.calibrateCamera([obj_points1] , [img_points], (640,480), cameraMatrix,None)

#rmtx = cv2.Rodrigues(rotation)[0]
#print(type(rotation),rotation,type(translation),translation)
projMtx = getProjMtx(np.array(rotation[0]),np.array(translation[0]))
print("projMtx: ",projMtx)

_3d = np.array([3,3,0,1])
_2d = np.dot(np.dot(cameraMatrix,projMtx,),_3d)
_2d = _2d/_2d[2]
print("2d cord",_2d)