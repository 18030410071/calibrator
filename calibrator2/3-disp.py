import cv2
import glob
import numpy as np
import shelve
from loader import *
import matplotlib.pyplot as plt



cameraMatrix1 = load_camera_matrix()
cameraMatrix2 = load_camera_matrix_r()
distCoeffs1 = load_camera_distortion()
distCoeffs2 = load_camera_distortion_r()

rvecs = None
tvecs = None

left_map1 = None
right_map1 = None

P1 = load_camera_matrix_proj()
P2 = load_camera_matrix_proj_r()
R1 = load_camera_matrix_rot()
R2 = load_camera_matrix_rot_r()
Q = load_camera_matrix_q()


class Config(object):
    sample = '01'  # 测试图片
    disp_calib = False  # 是否展示单目校正结果
    stereo_calib = True  # 是否进行双目校正
    disp_stereo_calib = True  # 是否展示双目校正结果
    disparity = True  # 是否利用视差估算距离
    num = 3  # StereoSGBM_create 函数参数：最小可能的差异值
    blockSize = 5  # StereoSGBM_create 函数参数：匹配的块大小。


opt = Config()

'''
# some ERROR 
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
    (640, 480), R, T)
'''

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


    images = glob.glob('stereo512/left.bmp')
    images_r = glob.glob('stereo512/right.bmp')



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
        else:
            print("No Corners")

    ret, rotation, translation = cv2.solvePnP(objpoints[0], imgpoints[0], 
    cameraMatrix1, dist)

    rvecs = [rotation]
    tvecs = [translation]

    left_map1, left_map2 = cv2.initUndistortRectifyMap(cameraMatrix1,
                                                       distCoeffs1, R1, P1,
                                                       gray.shape[::-1],
                                                       cv2.INTER_NEAREST)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(cameraMatrix2,
                                                         distCoeffs2, R2,
                                                         P2, gray.shape[::-1],
                                                         cv2.INTER_NEAREST)

    print("map:",left_map1.shape, left_map2.shape)
    print(left_map1[394][539][0])
    print(left_map1[394][539][1])

    print(left_map1[340][199][0])
    print(left_map1[340][199][1])

    print(right_map1[393][511][0])
    print(right_map1[393][511][1])

    print(right_map1[143][228][0])
    print(right_map1[143][228][1])

    img = cv2.imread('stereo512/left.bmp')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.imread('stereo512/right.bmp')

    gray_r = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #矫正原始图像 矫正后基本处于行对准状态
    imgL = cv2.remap(gray, left_map1, left_map2, cv2.INTER_LINEAR)
    imgR = cv2.remap(gray_r, right_map1, right_map2, cv2.INTER_LINEAR)

    if opt.disp_stereo_calib:
        cv2.imwrite(
            'result-stereo_calibresult-left' + str(opt.sample) + '.png',
            imgL)
        cv2.imwrite(
            'result-stereo_calibresult-right' + str(opt.sample) + '.png',
            imgR)

        plt.subplot(121)
        plt.title('left')
        plt.imshow(imgL, cmap='gray')
        plt.axis('off')
        plt.subplot(122)
        plt.title('right')
        plt.imshow(imgR, cmap='gray')
        plt.axis('off')
        plt.show()

    if not opt.disparity:
        exit(0)

    cv2.namedWindow("depth")
    cv2.namedWindow("disparity")
    cv2.moveWindow("depth", 0, 0)
    cv2.moveWindow("disparity", 600, 0)

    def callbackFunc(e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            print(threeD[y][x])

    cv2.setMouseCallback("depth", callbackFunc, None)

    stereo = cv2.StereoSGBM_create(numDisparities=16 * opt.num,
                                   blockSize=opt.blockSize)
    disparity = stereo.compute(imgL, imgR)

    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255,
                         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 将图片扩展至3d空间中，其z方向的值则为当前的距离
    threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16., Q)
    print(threeD.shape)

    '''

    cv2.imshow("disparity", disp)
    cv2.imshow("depth", imgL)

    key = cv2.waitKey(0)
    if key == ord("q"):
        exit(0)
    elif key == ord("s"):
        cv2.imwrite("result-disparity-disparity"+opt.sample+".png", disp)
    '''

def getSrcCoor(_map, x, y):
    return _map[y][x][0], _map[y][x][1]

def getImagePoints():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2) 

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



if __name__ == '__main__':
    calibration()




