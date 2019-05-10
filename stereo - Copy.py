import numpy as np
import cv2
import os
import util.geometry as ge
import glob
import shelve
from loader import *

_calibrator_path_left = "./data/paramleft"
_calibrator_path_right = "./data/paramright"

# 内参
_camera_matrix_left = load_camera_matrix(_calibrator_path_left)
_camera_distortion_left = load_camera_distortion(_calibrator_path_left)
_camera_tuned_matrix_left = load_camera_matrix_tuned(_calibrator_path_left)

_camera_matrix_right = load_camera_matrix(_calibrator_path_right)
_camera_distortion_right = load_camera_distortion(_calibrator_path_right)
_camera_tuned_matrix_right = load_camera_matrix_tuned(_calibrator_path_right)

print(_camera_tuned_matrix_left,"\n",_camera_tuned_matrix_right)
print(_camera_matrix_left,"\n",_camera_matrix_right)

#

def getProjMtx(rvec,tvec):
    rmtx, _ = cv2.Rodrigues(rvec)
    return np.hstack((rmtx,tvec))

def _calibrate_camera(path,tuned_matrix,distor):
    """
    generate calibration matrix ad distortion
    :return: calibration matrix and distortion
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    vertical, horizon = 7, 9  # target corners in vertical and horizontal direction

    grid = np.zeros((vertical * horizon, 3), np.float32)
    grid[:, :2] = np.mgrid[:horizon, :vertical].T.reshape(-1, 2)

    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane

    image_list = glob.glob(os.path.join(path, "*.bmp"))
    #print(image_list)
    gray = None
    for img_name in image_list:
        print(img_name)
        image = cv2.imread(img_name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # find the chess board corners
        found, corners = cv2.findChessboardCorners(gray, (horizon, vertical), None)
        #print("corner shape",corners.shape,corners)

        # add object points, image points (after refining them)
        if found:
            obj_points.append(grid)
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  #corner是像素坐标 refine 坐标
            img_points.append(corners)
            #print("corner shape",corners.shape,corners)
        else:
            print('can not find %s corners' % img_name)

    #print(corners)
    #print(grid)
    #print("before tuned_matrix\n",_camera_tuned_matrix_right,"\n",_camera_tuned_matrix_left)
    ret, rotation, translation = cv2.solvePnP(obj_points[0], img_points[0], tuned_matrix, distor)

    #ret, matrix, distortion, rotation, translation = \
        #cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], tuned_matrix, distor,)
        
        #cv2.calibrateCamera([obj_points[-1]], [img_points[-1]], gray.shape[::-1], None, None)
    #print("RT Matrix",np.array(rotation).shape,np.array(translation).shape,rotation)
    #print("after tuned_matrix",_camera_tuned_matrix_right,"\n",_camera_tuned_matrix_left)
    #print("rtMtx:\n",getProjMtx(np.array(rotation[0]),np.array(translation[0])))
    #rtMtx = getProjMtx(np.array(rotation[0]),np.array(translation[0]))
    rtMtx = getProjMtx(rotation,translation)
    _3d = np.array([8,0,0,1])
    _2d = np.dot(np.dot(tuned_matrix,rtMtx,),_3d)
    _2d = _2d/_2d[2]
    print("2d-src:",_2d,img_points[0][7])
    mean_error = 0
    for i in range(len(obj_points)):
        new_img_points, _ = cv2.projectPoints(obj_points[i], rotation, translation, tuned_matrix, distor)#3D点投影到平面
        error = cv2.norm(img_points[i], new_img_points, cv2.NORM_L2) / len(new_img_points)
        mean_error += error
        print("cv-proj and old:",new_img_points[1],img_points[i][1])
    print("mean error: ", mean_error / len(obj_points))

    #return np.array(matrix), np.array(distortion)
    return tuned_matrix,rtMtx

matrixl, rtMtxl = _calibrate_camera("C:\\Users\\chuyangl\\Desktop\\liushuai\\calibrator\\board\\left_S",
                                    _camera_matrix_left,_camera_distortion_left)
matrixr, rtMtxr = _calibrate_camera("C:\\Users\\chuyangl\\Desktop\\liushuai\\calibrator\\board\\right_S",
                                    _camera_matrix_right,_camera_distortion_right)  
#print("matrix_tuned\n",_camera_tuned_matrix_right)
#print("matrixr\n",matrixr)
projl = np.dot(matrixl,rtMtxl)
projr = np.dot(matrixr,rtMtxr)
#print(projl,projr)
_3dl = np.array([4.2,0.8,0,1])
_2dl = np.dot(projl,_3dl)
_2dl = _2dl/_2dl[2]
_2dl = np.array([_2dl]).T

_3dr = np.array([2.7,0.7,0,1])
_2dr = np.dot(projr,_3dr)
_2dr = _2dr/_2dr[2]
_2dr = np.array([_2dr]).T
print("_2dl,_2dr",_2dl,"\n", _2dr)

# The cv2 method
#X = cv2.triangulatePoints( projl, projr, a3xN[:2], b3xN[:2] )  # coor 
_2dl = np.array([[498.0, 406., 447.],
                [186.0, 180. ,180.],
                [1., 1. ,1.]])
_2dr = np.array([[130.0, 70., 96],
                [179.0, 168.,170.],
                [1., 1., 1.]])


X = cv2.triangulatePoints( projl, projr, _2dl[:2], _2dr[:2] )  # coor 
# Remember to divide out the 4th row. Make it homogeneous
X /= X[3]
# Recover the origin arrays from PX
x1 = np.dot(projl,X)
x2 = np.dot(projr,X)
# Again, put in homogeneous form before using them
x1 /= x1[2]
x2 /= x2[2]
 
print('X\n',X)
print('x1\n',x1)
print('x2\n',x2)

