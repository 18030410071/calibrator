import numpy as np
import cv2
import os
import util.geometry as ge
import glob
import shelve
from loader import *

'''
_camera_matrix = load_camera_matrix()
_camera_distortion = load_camera_distortion()
_camera_tuned_matrix = load_camera_matrix_tuned()

_remap_x, _remap_y = cv2.initUndistortRectifyMap(_camera_matrix, _camera_distortion, None, _camera_tuned_matrix,
                                              (CAMERA_WIDTH, CAMERA_HEIGHT), cv2.CV_32FC1)
'''

def un_distort_image(image):
    """
    un_distort an image by #remap# api, faster than #undistort# api
    :param image: image to process
    :return: undistorted image
    """
    global _remap_x, _remap_y
    image = cv2.UMat(image)
    res = cv2.remap(image, _remap_x, _remap_y, cv2.INTER_LINEAR)   # 进行remap
    res = res.get()
    return res


def un_distort_point(point):
    """
    un_distort a specific point
    :param point: point to distort
    :return: undistorted point
    """
    points = np.array([[(point.x, point.y)]], np.float32)
    temp = cv2.undistortPoints(points, _camera_matrix, _camera_distortion)
    fx, fy = _camera_tuned_matrix[0][0], _camera_tuned_matrix[1][1]
    cx, cy = _camera_tuned_matrix[0][2], _camera_tuned_matrix[1][2]
    x = temp[0][0][0] * fx + cx
    y = temp[0][0][1] * fy + cy
    return ge.Point(x, y)


def distort_point(point):
    """
    distort a specific point
    :param point:  point to distort
    :return: distorted point
    """
    fx, fy = _camera_tuned_matrix[0][0], _camera_tuned_matrix[1][1]
    cx, cy = _camera_tuned_matrix[0][2], _camera_tuned_matrix[1][2]
    x, y = (point.x - cx) / fx, (point.y - cy) / fy

    k1, k2, p1, p2, k3 = _camera_distortion[0]
    r2 = x ** 2 + y ** 2
    r4 = r2 * r2
    r6 = r2 * r4
    x = x * (1 + k1 * r2 + k2 * r4 + k3 * r6) + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    y = y * (1 + k1 * r2 + k2 * r4 + k3 * r6) + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

    fx2, fy2 = _camera_matrix[0][0], _camera_matrix[1][1]
    cx2, cy2 = _camera_matrix[0][2], _camera_matrix[1][2]
    x2 = x * fx2 + cx2
    y2 = y * fy2 + cy2
    return ge.Point(x2, y2)

def getProjMtx(rvec,tvec):
    rmtx = cv2.Rodrigues(rvec)[0]
    return np.hstack((rmtx,tvec))


def generate_chessboard():
    """
    generate chessboard for calibration
    """
    width = 2105
    height = 1487
    chessboard = np.zeros((height, width, 3), np.uint8)

    cell_size = 200
    shift = 50
    for r in list(range(height)):
        for c in list(range(width)):
            if (int(r / cell_size) + int((c - shift) / cell_size)) % 2 == 1:
                chessboard[r, c, :] = [255, 255, 255]

    #cv2.imshow('chessboard', cv2.resize(chessboard, (int(width / 3), int(height / 3))))
    full_path = os.path.join('calibration', 'chessboard.png')
    cv2.imwrite(full_path, chessboard)
    cv2.waitKey(0)
    return chessboard


def _calibrate_camera():
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

    image_list = glob.glob(os.path.join('C:\\Users\\chuyangl\\Desktop\\liushuai\\calibrator\\board\\right', "*.bmp"))
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
    ret, matrix, distortion, rotation, translation = \
        cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    print(type(img_points[0]),img_points[0].shape,img_points[0])
    print(type(obj_points[0]),obj_points[0].shape,obj_points[0])    
        #cv2.calibrateCamera([obj_points[-1]], [img_points[-1]], gray.shape[::-1], None, None)
    #print("RT Matrix",np.array(rotation).shape,np.array(translation).shape,rotation)
    
    print("projMtx:\n",getProjMtx(np.array(rotation[0]),np.array(translation[0])))
    projMtx = getProjMtx(np.array(rotation[0]),np.array(translation[0]))
    _3d = np.array([8,0,0,1])
    _2d = np.dot(np.dot(matrix,projMtx,),_3d)
    _2d = _2d/_2d[2]
    print("2d:",_2d,img_points[0][7])
    mean_error = 0
    for i in range(len(obj_points)):
        new_img_points, _ = cv2.projectPoints(obj_points[i], rotation[i], translation[i], matrix, distortion)#3D点投影到平面
        error = cv2.norm(img_points[i], new_img_points, cv2.NORM_L2) / len(new_img_points)
        mean_error += error
        print("new and old:",new_img_points[1],img_points[i][1])
    print("mean error: ", mean_error / len(obj_points))

    return np.array(matrix), np.array(distortion)


def _check_calibration():
    """
    check if the calibration is correct
    """
    image_list = glob.glob(os.path.join("C:\\Users\\chuyangl\\Desktop\\liushuai\\calibrator\\board\\left", "*.bmp"))
    for single_img in image_list:
        image = cv2.imread(single_img)
        new_image = un_distort_image(image)
        cv2.imshow('before', cv2.resize(image, (int(image.shape[1] * 0.7), int(image.shape[0] * 0.7))))
        cv2.imshow('after', cv2.resize(new_image, (int(new_image.shape[1] * 0.7), int(new_image.shape[0] * 0.7))))
        cv2.waitKey(0)

    image = cv2.imread(image_list[0])

    # distortion_points = [ge.Point(110, 437), ge.Point(932, 151), ge.Point(1034, 331)]
    # calibration_points = [ge.Point(510, 437), ge.Point(832, 151), ge.Point(1134, 331)]

    distortion_points = [ge.Point(110, 437), ge.Point(632, 151), ge.Point(333, 331)]
    calibration_points = [ge.Point(510, 437), ge.Point(532, 151), ge.Point(234, 331)]

    for p in distortion_points:
        cv2.circle(image, p.tuple(), 23, (0, 0, 255), 2)

    new_image = un_distort_image(image)

    for p in calibration_points:
        cv2.circle(new_image, p.tuple(), 23, (255, 0, 0), 4)
        p2 = distort_point(p)
        p3 = un_distort_point(p2)
        cv2.circle(image, p2.int().tuple(), 23, (0, 255, 255), 4)
        cv2.circle(new_image, p3.int().tuple(), 23, (0, 0, 255), 4)
        print(p.int().tuple(), p2.int().tuple(), p3.int().tuple())

    for p in distortion_points:
        p2 = un_distort_point(p)
        p3 = distort_point(p2)
        cv2.circle(new_image, p2.int().tuple(), 23, (0, 255, 255), 2)
        cv2.circle(image, p3.int().tuple(), 23, (0, 255, 255), 2)
        print(p.int().tuple(), p2.int().tuple(), p3.int().tuple())

    cv2.imshow('before', cv2.resize(image, (int(image.shape[1] * 0.7), int(image.shape[0] * 0.7))))
    cv2.imshow('after', cv2.resize(new_image, (int(new_image.shape[1] * 0.7), int(new_image.shape[0] * 0.7))))

    cv2.waitKey(0)


if __name__ == "__main__":
    
    
    pos = generate_chessboard()
    _camera_matrix, _camera_distortion = _calibrate_camera()

    save_camera_matrix(_camera_matrix)
    print("camera_matrix:\n",_camera_matrix)
    print("distortion:\n",_camera_distortion)
    _camera_tuned_matrix, _ = cv2.getOptimalNewCameraMatrix(_camera_matrix, _camera_distortion, (CAMERA_WIDTH, CAMERA_HEIGHT), 1, (CAMERA_WIDTH, CAMERA_HEIGHT))
    print("camera_tuned_matrix:\n",_camera_tuned_matrix)

    save_camera_matrix_tuned(_camera_tuned_matrix)
    save_camera_distortion(_camera_distortion)
    print("matrix and distortion saved.")
    


    #_check_calibration()

