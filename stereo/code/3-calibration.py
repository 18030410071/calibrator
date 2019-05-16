import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

import fire

data_in = np.load("in_cal.npz")
mtx = data_in["mtx"]
dist = data_in["dist"]
mtx_r = data_in["mtx_r"]
dist_r = data_in["dist_r"]

rvecs = None
tvecs = None

cameraMatrix1 = mtx
cameraMatrix2 = mtx_r

data_ex = np.load("ex_cal.npz")
P1 = data_ex["P1"]
P2 = data_ex["P2"]
R1 = data_ex["R1"]
R2 = data_ex["R2"]
Q = data_ex["Q"]

def calibration():
    global rvecs, tvecs
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2) * 25

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    objpoints_r = []
    imgpoints_r = []


    images = glob.glob('../stereo512/left.bmp')
    images_r = glob.glob('../stereo512/right.bmp')



    for fname, fname_r in zip(images, images_r):

        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_r = cv2.imread(fname_r)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        
        left_map1, left_map2 = cv2.initUndistortRectifyMap(cameraMatrix1,
                                                        dist, R1, P1,
                                                        gray.shape[::-1],
                                                        cv2.INTER_NEAREST)
        right_map1, right_map2 = cv2.initUndistortRectifyMap(cameraMatrix2,
                                                            dist_r, R2,
                                                            P2, gray_r.shape[::-1],
                                                            cv2.INTER_NEAREST)

        imgL = cv2.remap(gray, left_map1, left_map2, cv2.INTER_LINEAR)
        imgR = cv2.remap(gray_r, right_map1, right_map2, cv2.INTER_LINEAR)
        cv2.imwrite("left-remap.jpg", imgL)
        cv2.imwrite("right-remap.jpg",imgR)

        gray = imgL
        gray_r = imgR
        
        stereo = cv2.StereoSGBM_create(numDisparities=48,
                                   blockSize=5)
        disparity = stereo.compute(imgL, imgR)

        disp = cv2.normalize(disparity, disparity, alpha=0, beta=255,
                         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 将图片扩展至3d空间中，其z方向的值则为当前的距离
        threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16., Q)
        print("ThreeD:", threeD.shape,threeD[:,:,-1])
        
    
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 7), None)
        #print('corners',corners)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 7), None)

        # If found, add object points, image points (after refining them)
        if ret == True and ret_r == True:
            objpoints.append(objp)
            objpoints_r.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria)
            corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1),
                                          criteria)
            imgpoints.append(corners2)
            imgpoints_r.append(corners2_r)
        else:
            print("NO Corners")

    ret, rotation, translation = cv2.solvePnP(objpoints[0], imgpoints[0], 
    cameraMatrix1, dist)

    rvecs = [rotation]
    tvecs = [translation]
    print("r t vecs:",rvecs,tvecs)

    l = imgpoints[0].reshape(63,2).T
    r = imgpoints_r[0].reshape(63,2).T

    print("left cam pixel", l.shape, l)
    print("right cam pixel", r.shape, r) 
    p4d = cv2.triangulatePoints(P1, P2, l, r)
    print("left camear p4d\n", p4d/p4d[-1])
    X = p4d/p4d[-1]
    # Recover the origin arrays from PX
    x1 = np.dot(P1,X)
    x2 = np.dot(P2,X)
    # Again, put in homogeneous form before using them
    #x1 /= x1[2]
    #x2 /= x2[2]
    
    #print('x1\n',x1, "\nsrc-l\n", imgpoints[0])
    #print('x2\n',x2, "\nsrc-r\n", imgpoints_r[0])

    rmtx, _ = cv2.Rodrigues(rvecs[0]) #
    rtmtxl = np.hstack((rmtx, tvecs[0]))

    rtmtxl_homo = np.vstack((rtmtxl,np.array([0,0,0,1])))
    obj_homo = cv2.convertPointsToHomogeneous(objpoints[0]).reshape(63,4).T
    print("obj_homo", obj_homo)
    #print(rtmtxl_homo, obj_homo)
    P_R_T = np.dot(rmtx, objpoints[0].T) + tvecs[0]
    print("P*R + T\n", P_R_T)
    print("P*RT:\n", np.dot(rtmtxl_homo, obj_homo))
    print("P*RT/p4d\n", np.dot(rtmtxl_homo, obj_homo)/X)

    u_v= np.dot(cameraMatrix1, P_R_T)

    print("u_v/x1\n", u_v/x1) 

    # 使用逆求棋盘格3d坐标
    mat = np.mat(rtmtxl).I
    rtmtxl_I = np.array(mat) #棋盘->左相机 RT的逆

    print(rtmtxl_I.shape)
    #tmpl = np.dot(rtmtxl_I.T, p4d)
    tmpl = np.dot(np.dot(np.dot(rtmtxl_I, np.linalg.pinv(cameraMatrix1)), P1),p4d)
    print("chessboard p4d tmpl\n", tmpl.shape, "\n", tmpl/tmpl[-1])
    print("average:", np.sum(tmpl[:-2])/63)
    ###
    #print('stereoRec',R1)
    #print('stereoRec',R2)
    print('stereoRec P1\n',P1)
    print('l-stere-mtx\n',cameraMatrix1,"\nmtx\n",mtx)
    print('stereoRec P2\n',P2)
    #print('Q',Q)
'''
    undistortImage("F:\\myrepo\\calibrator\\postgraduate-master\\right\\right01.jpg", 
                   "F:\\myrepo\\calibrator\\postgraduate-master\\right\\right01_undistor.jpg", 
               cameraMatrix1, distCoeffs1)
import os
def undistortImage(filename, savename, _cam_mtx, _cam_dis):
    image = cv2.imread(filename)
    new_image = cv2.undistort(image, _cam_mtx, _cam_dis)
    cv2.imwrite(savename, new_image)
'''
def getImagePoints():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2) 
    #print(objp.shape)
    #objp[:, -1] = 0
    #print(objp)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    objpoints_r = []
    imgpoints_r = []

    #images = glob.glob('../left/left01_undistor.jpg')
    #images_r = glob.glob('../right/right01_undistor.jpg')
    images = glob.glob('../stereo512/left.bmp')
    images_r = glob.glob('../stereo512/right.bmp')

    images.sort()
    images_r.sort()

    for fname, fname_r in zip(images, images_r):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_r = cv2.imread(fname_r)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 7), None)
        #print('corners',corners)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 7), None)

        # If found, add object points, image points (after refining them)
        if ret == True and ret_r == True:
            objpoints.append(objp)
            objpoints_r.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria)
            corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1),
                                          criteria)
            imgpoints.append(corners2)
            imgpoints_r.append(corners2_r)
    l = imgpoints[0].reshape(63, 2).T
    r = imgpoints_r[0].reshape(63, 2).T

    return l, r

def undistortImage(img,_cam_mtx, _cam_dis):
    new_image = cv2.undistort(img, _cam_mtx, _cam_dis)
    return new_image



def getp3d(imgpoints_l, imgpoints_r):
    global rvecs, tvecs
    '''
    l : left  cam imgpoints  2 * N [[x1, x2,...xn], [y1, y2,...yn]]
    r : right cam imgpoints  2 * N
    return : 3 * N  [[x1...xn], [y1...yn], [z1...zn]]
    '''
    l = imgpoints_l
    r = imgpoints_r

    p4d = cv2.triangulatePoints(P1, P2, l, r)

    X = p4d/p4d[-1]  # 3d in left cam


    rmtx, _ = cv2.Rodrigues(rvecs[0]) #
    rtmtxl = np.hstack((rmtx, tvecs[0]))

    # 使用逆求棋盘格3d坐标
    mat = np.mat(rtmtxl).I
    rtmtxl_I = np.array(mat) #棋盘->左相机 RT的逆

    _3dl = np.dot(np.dot(np.dot(rtmtxl_I, np.linalg.pinv(cameraMatrix1)), P1),p4d)

    return _3dl/_3dl[-1]
    


if __name__ == '__main__':
    calibration()
    #l, r = getImagePoints()
    #p3d = getp3d(l, r)
    l = np.array([[432.0, 378.0], [244.0, 168.0]])
    r = np.array([[283.0, 207.0], [249.0, 164.0]])
    p3d = getp3d(l, r)
    print("p3d:")
    print(p3d)