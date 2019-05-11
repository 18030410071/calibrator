import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

import fire



def calibration():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2) * 100
    #print(objp.shape)
    #objp[:, -1] = 0
    #print(objp)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    objpoints_r = []
    imgpoints_r = []

    images = glob.glob('../left/left01.jpg')
    images_r = glob.glob('../right/right01.jpg')
    images.sort()
    images_r.sort()

    for fname, fname_r in zip(images, images_r):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_r = cv2.imread(fname_r)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
        #print('corners',corners)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, (7, 6), None)

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
    data = np.load("in_cal.npz")
    mtx = data["mtx"]
    dist = data["dist"]
    mtx_r = data["mtx_r"]
    dist_r = data["dist_r"]

    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
        cv2.stereoCalibrate([objpoints[0]], [imgpoints[0]], [imgpoints_r[0]], mtx,
                            dist, mtx_r, dist_r, gray.shape[::-1])

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
        gray.shape[::-1], R, T)

    np.savez("ex_cal.npz", P1=P1, P2=P2)
if __name__ == '__main__':
    fire.Fire(calibration)
    print("ints me!")