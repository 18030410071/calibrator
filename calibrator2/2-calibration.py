import cv2
import glob
import numpy as np
import shelve
from loader import *

mtx = load_camera_matrix()
mtx_r = load_camera_matrix_r()
dist = load_camera_distortion()
dist_r = load_camera_distortion_r()


def undistortImage(img,_cam_mtx, _cam_dis):
    new_image = cv2.undistort(img, _cam_mtx, _cam_dis)
    return new_image

def calibration(undistort=False):
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
            img = undistortImage(img, mtx, dist)
            img_r = undistortImage(img_r, mtx_r, dist_r)    
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

            # Draw and display the corners
            # cv2.imshow('img', img)
            # cv2.waitKey(500)

    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
    cv2.stereoCalibrate(objpoints, imgpoints, imgpoints_r, mtx,
                        dist, mtx_r, dist_r, gray.shape[::-1], flags = cv2.CALIB_FIX_INTRINSIC )
    print("OPENCV R-T\n",R, "\n", T)

    '''matlab
    R = np.array([[0.898066037373019,	0.0213408993926283,	-0.439342643650985],
                [-0.0214251464935368,	0.999759088048663,	0.00476749009820505],
                [0.439338543283939,	0.00513145956036998,	0.898306914427317]])

    T = np.array([-264.886066592313,	-1.77392898927413,	46.7689011903979])

    print("MATLAB R_T",R,"\n",T)
    matlab'''

    
    print("mtx\n", mtx)
    print("mtx_r", mtx_r)
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
        gray.shape[::-1], R, T, flags = 0)
    print("P1,P2\n",P1,"\n", P2)
    #np.savez("ex_cal.npz", P1=P1, P2=P2, R1=R1, R2=R2, Q=Q)
    save_camera_matrix_rot(R1)
    save_camera_matrix_rot_r(R2)
    save_camera_matrix_proj(P1)
    save_camera_matrix_proj_r(P2)
    save_camera_matrix_q(Q)
if __name__ == '__main__':
    calibration(undistort=True)
    print("External Done!")