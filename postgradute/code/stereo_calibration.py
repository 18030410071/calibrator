import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

import fire



def calibration(des='01', data_file='../stat/matlab.mat'):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2) 
    #print(objp.shape)
    #objp[:, -1] = 0
    #print(objp)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    objpoints_r = []
    imgpoints_r = []

    images = glob.glob('../left/*.jpg')
    images_r = glob.glob('../right/*.jpg')
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

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                       gray.shape[::-1], None,
                                                       None)
    img = cv2.imread('../left/left' + str(des) + '.jpg')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1,
                                                      (w, h))
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]

    ret, mtx_r, dist_r, rvecs, tvecs = cv2.calibrateCamera(objpoints_r,
                                                           imgpoints_r,
                                                           gray_r.shape[::-1],
                                                           None, None)
    img_r = cv2.imread('../right/right' + str(des) + '.jpg')
    h, w = img_r.shape[:2]
    newcameramtx_r, roi = cv2.getOptimalNewCameraMatrix(mtx_r, dist_r, (w, h),
                                                        1, (w, h))
    # undistort
    dst_r = cv2.undistort(img_r, mtx_r, dist_r, None, newcameramtx_r)

    # crop the image
    x, y, w, h = roi
    dst_r = dst_r[y:y + h, x:x + w]

    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
        cv2.stereoCalibrate(objpoints, imgpoints, imgpoints_r, mtx,
                            dist, mtx_r, dist_r, gray.shape[::-1])
    #print('lcmx',cameraMatrix1,"\n",mtx)
    print('rcmx',cameraMatrix2,"\n",mtx_r)
    print('dis coef',dist,"\n",distCoeffs1)
    R = np.array([[1, -0.0032, -0.005], [0.0033, 0.9999, 0.0096],
                  [0.0057, -0.0097, 0.9999]])
    T = np.array([-83.0973, 1.0605, 0.0392])
    # TODO: import mat and read stat from mat file.

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
        gray.shape[::-1], R, T)
    #print('stereoRec P1',P1)
    ### add triangulatePoints()
    l = cv2.convertPointsToHomogeneous(imgpoints[0]).reshape(42, 3).T[:2]
    r = cv2.convertPointsToHomogeneous(imgpoints_r[0]).reshape(42, 3).T[:2]
    print("left cam pixel",l.shape, l)
    p4d = cv2.triangulatePoints(P1, P2, l, r)
    print("left camear p4d\n", p4d/p4d[-1])
    X = p4d/p4d[-1]
    # Recover the origin arrays from PX
    x1 = np.dot(P1,X)
    x2 = np.dot(P2,X)
    # Again, put in homogeneous form before using them
    x1 /= x1[2]
    x2 /= x2[2]
    
    print('x1\n',x1, "\nsrc-l\n", imgpoints[0])
    print('x2\n',x2, "\nsrc-r\n", imgpoints_r[0])

    rmtx, _ = cv2.Rodrigues(rvecs[0])
    print(rmtx, tvecs[0])
    rtmtxl = np.hstack((rmtx, tvecs[0]))
    mat = np.mat(rtmtxl).I
    rtmtxl_I = np.array(mat)


    print(rtmtxl_I.shape)
    #tmpl = np.dot(rtmtxl_I.T, p4d)
    tmpl = np.dot(np.dot(np.dot(rtmtxl_I, np.linalg.pinv(cameraMatrix1)), P1),p4d)
    print("chessboard p4d tmpl\n", tmpl.shape, "\n", tmpl/tmpl[-1])
    ###
    #print('stereoRec',R1)
    #print('stereoRec',R2)
    print('stereoRec P1\n',P1)
    print('lcmx\n',cameraMatrix1,"\nmtx\n",mtx)
    print('stereoRec P2\n',P2)
    #print('Q',Q)
    left_map1, left_map2 = cv2.initUndistortRectifyMap(cameraMatrix1,
                                                       distCoeffs1, R1, P1,
                                                       gray.shape[::-1],
                                                       cv2.INTER_NEAREST)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(cameraMatrix2,
                                                         distCoeffs2, R2,
                                                         P2, gray.shape[::-1],
                                                         cv2.INTER_NEAREST)

    img = cv2.imread('../left/left' + str(des) + '.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.imread(('../right/right' + str(des) + '.jpg'))
    gray_r = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    des_l = cv2.remap(gray, left_map1, left_map2, cv2.INTER_LINEAR)
    cv2.imwrite('../result/stereo_calibresult/left' + str(des) + '.png', des_l)
    des_r = cv2.remap(gray_r, right_map1, right_map2, cv2.INTER_LINEAR)
    cv2.imwrite('../result/stereo_calibresult/right' + str(des) + '.png', des_r)

    plt.subplot(121)
    plt.title('left')
    plt.imshow(des_l, cmap='gray')
    plt.axis('off')
    plt.subplot(122)
    plt.title('right')
    plt.imshow(des_r, cmap='gray')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    fire.Fire(calibration)
