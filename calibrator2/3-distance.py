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

P1_stereo = load_camera_matrix_stereo_proj()
P2_stereo = load_camera_matrix_stereo_proj_r()
P1_own = load_camera_matrix_own_proj()
P2_own = load_camera_matrix_own_proj_r()

R1 = load_camera_matrix_rot()
R2 = load_camera_matrix_rot_r()
Q = load_camera_matrix_q()

GRAY_SHAPE = (640, 480)
left_map1, left_map2 = cv2.initUndistortRectifyMap(cameraMatrix1,
                                                    distCoeffs1, R1, P1_stereo,
                                                    GRAY_SHAPE,
                                                    cv2.INTER_NEAREST)
right_map1, right_map2 = cv2.initUndistortRectifyMap(cameraMatrix2,
                                                    distCoeffs2, R2, P2_stereo, 
                                                    GRAY_SHAPE,
                                                    cv2.INTER_NEAREST)

def getAlignImage(gray, gray_r):
    """
    获得行对齐的图片
    输入: 左 右 相机的灰度图 gray, gray_r
    输出: 行对齐的左右相机视图
    """
    imgL = cv2.remap(gray, left_map1, left_map2, cv2.INTER_LINEAR)
    imgR = cv2.remap(gray_r, right_map1, right_map2, cv2.INTER_LINEAR)
    return imgL, imgR

def getProjCoorL(xl, yl):
    """
    获得左相机行对齐图片里物体像素在未对齐图片里的位置
    输入: 在行对齐的左相机里的像素坐标 x, y
    输出: 对应的未对齐的图片里的像素坐标
    """
    return (float(left_map1[yl][xl][0]), float(left_map1[yl][xl][1]))

def getProjCoorR(xr, yr):
    """
    获得左相机行对齐图片里物体像素在未对齐图片里的位置
    输入: 在行对齐的左相机里的像素坐标 x ,y 
    输出: 对应的未对齐的图片里的像素坐标
    """
    return (float(right_map1[yr][xr][0]), float(right_map1[yr][xr][1]))

def getp3d(imgpoints_l, imgpoints_r):
    """
    获得目标的3D坐标 世界坐标系为棋盘或者桌面所在平面为xoy面的笛卡尔直角坐标系
    输入: 左相机视图的物体坐标 格式 [[x1, x2,...xn], [y1, y2,...yn]]
          右相机视图的物体坐标 格式 [[x1, x2,...xn], [y1, y2,...yn]]
    输出: 目标的3D坐标    格式 [[x1...xn], [y1...yn], [z1...zn]]
    注: 输入必须是float型
    """
    global P1_own, P2_own
    l = imgpoints_l
    r = imgpoints_r

    p4d = cv2.triangulatePoints(P1_own, P2_own, l, r)
    X = p4d/p4d[-1]  # 3d in chessboard coor

    return X[:-1]


if __name__ == '__main__':
    # 读入左右相机原始图片
    img = cv2.imread('stereo512/left/Image_20190512100955943.bmp')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_r = cv2.imread('stereo512/right/Image_20190512100953583.bmp')
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    #进行行对准操作
    imgL, imgR = getAlignImage(gray, gray_r)

    image = np.hstack((imgL, imgR))
    image =  cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    point_color = (0, 255, 0) # BGR
    thickness = 1 
    lineType = 4
    for i in range(10):
        ptStart = (0, 48 * i)
        ptEnd = (1280, 48 * i)
        cv2.line(image, ptStart, ptEnd, point_color, thickness, lineType)
    # 找到行对准之后的左右点的坐标对 计算其对应的左右相机原始坐标
    xl, yl = getProjCoorL(455, 240)
    xr, yr = getProjCoorR(297, 240)

    # 计算3d坐标
    print(xl,yl,xr,yr)
    l = np.array([[xl],[yl]])
    r = np.array([[xr],[yr]])
    print("p3d:\n", getp3d(l, r))

    image_bottom = np.hstack((img, img_r))
    point_color = (255, 0, 0)
    for i in range(10):
        ptStart = (0, 48 * i)
        ptEnd = (1280, 48 * i)
        cv2.line(image_bottom, ptStart, ptEnd, point_color, thickness, lineType)

    image_FOUR = np.vstack((image,image_bottom))

    ptStart = (455, 240)
    ptEnd = (int(xl), int(yl) + 480)
    point_color = (0, 0, 255)
    cv2.line(image_FOUR, ptStart, ptEnd, point_color, thickness, lineType)


    ptStart = (297 + 640, 240)
    ptEnd = (640 + int(xr), int(yr)+480)
    point_color = (0, 0, 255)
    cv2.line(image_FOUR, ptStart, ptEnd, point_color, thickness, lineType)
    cv2.imwrite("image_FOUR.png", image_FOUR)

    #cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
    #cv2.imshow('img', image_FOUR)
    #cv2.waitKey(1)

    cv2.imwrite("left-remap.png", imgL)
    cv2.imwrite("right-remap.png", imgR)




