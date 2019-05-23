import cv2
import glob
import numpy as np
import shelve
from loader import *

cameraMatrix1 = load_camera_matrix()
cameraMatrix2 = load_camera_matrix_r()
distCoeffs1 = load_camera_distortion()
distCoeffs2 = load_camera_distortion_r()

#使用自己定义的坐标系统时 首先初始化以下参数。
#默认使用棋盘格所在平面为xoy面的笛卡尔直角坐标系统。原点在左上。
objpoints_selfcoor = None  # 3d point in real world space
imgpoints_selfcoor = None  # 2d points in image plane.
imgpoints_r_selfcoor = None

def undistortImage(img,_cam_mtx, _cam_dis):
    """
    图像矫正
    输入: 待矫正图像 内参矩阵 畸变系数
    输出: 矫正后的图像
    """
    new_image = cv2.undistort(img, _cam_mtx, _cam_dis)
    return new_image

def getrtMtx(rvec,tvec):
    """
    计算外参矩阵
    输入: 旋转向量 平移向量
    输出: 外参矩阵
    """
    rmtx, _ = cv2.Rodrigues(rvec)
    return np.hstack((rmtx,tvec))

def calibration(undistort=False, selfCoor=False):
    """
    主要完成以下功能:
        1. 分别计算单目的投影矩阵并持久化。为计算3d坐标做准备
        2. 计算双目的投影矩阵及R1 R2 Q 矩阵。为行对准做准备
    """
    global imgpoints_r_selfcoor, imgpoints_selfcoor, objpoints_selfcoor
    global cameraMatrix1, cameraMatrix2, distCoeffs1, distCoeffs2
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2) * 25
    #print(objp.shape)
    #objp[:, -1] = 0
    #print(objp)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    objpoints_r = []
    imgpoints_r = []

    images = glob.glob('stereo512/left.bmp')
    images_r = glob.glob('stereo512/right.bmp')
    images.sort()
    images_r.sort()

    for fname, fname_r in zip(images, images_r):
        img = cv2.imread(fname)
        img_r = cv2.imread(fname_r)

        if undistort:
            img = undistortImage(img, cameraMatrix1, distCoeffs1)
            img_r = undistortImage(img_r, cameraMatrix2, distCoeffs2)    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

    # 分别计算投影矩阵并持久化。
    if not selfCoor:  
        objpoints_selfcoor = objpoints[0]
        imgpoints_selfcoor = imgpoints[0]
        imgpoints_r_selfcoor = imgpoints_r[0]
    else:
        if objpoints_selfcoor == None or imgpoints_selfcoor == None or imgpoints_r_selfcoor == None:
            print("Initial the self-defined coordinate first")
            return
    ret, rotation, translation = cv2.solvePnP(objpoints_selfcoor, imgpoints_selfcoor, cameraMatrix1, distCoeffs1)

    ret, rotation_r, translation_r = cv2.solvePnP(objpoints_selfcoor, imgpoints_r_selfcoor, cameraMatrix2, distCoeffs2)

    rt1 = getrtMtx(rotation, translation)
    rt2 = getrtMtx(rotation_r, translation_r)

    P1_own = np.dot(cameraMatrix1, rt1)
    P2_own = np.dot(cameraMatrix2, rt2)
    save_camera_matrix_own_proj(P1_own)
    save_camera_matrix_own_proj_r(P2_own)

    # 双目计算 R1 R2 P1 P2 Q并持久化。
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
    cv2.stereoCalibrate(objpoints, imgpoints, imgpoints_r, cameraMatrix1,
                        distCoeffs1, cameraMatrix2, distCoeffs2, gray.shape[::-1], flags = cv2.CALIB_FIX_INTRINSIC )
    #print("OPENCV R-T\n",R, "\n", T)
    R1, R2, P1_stereo, P2_stereo, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
        gray.shape[::-1], R, T, flags = 0)
    #print("P1,P2\n",P1_stereo,"\n", P2_stereo)
    save_camera_matrix_rot(R1)
    save_camera_matrix_rot_r(R2)
    save_camera_matrix_stereo_proj(P1_stereo)
    save_camera_matrix_stereo_proj_r(P2_stereo)
    save_camera_matrix_q(Q)
if __name__ == '__main__':
    calibration(undistort=True,selfCoor=False)
    print("External Done!")