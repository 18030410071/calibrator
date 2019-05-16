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
    objp = np.zeros((7 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2) * 25

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    objpoints_r = []
    imgpoints_r = []

    images = glob.glob('../left512/*.bmp')
    images_r = glob.glob('../right512/*.bmp')
    #print(images,images_r)
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

            # Draw and display the corners
            # cv2.imshow('img', img)
            # cv2.waitKey(500)
        else:
            print("No corners found")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                       gray.shape[::-1], None,
                                                       None)

    ret, mtx_r, dist_r, rvecs, tvecs = cv2.calibrateCamera(objpoints_r,
                                                           imgpoints_r,
                                                           gray_r.shape[::-1],
                                                           None, None)
    np.savez("in_cal.npz", mtx=mtx,dist=dist, mtx_r=mtx_r,dist_r=dist_r)
    print("Intrinsic Done.")

if __name__ == '__main__':
    calibration()
