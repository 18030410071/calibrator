import cv2
import numpy as np
import glob
import os

def getProjMtx(rvec,tvec):
    rmtx, _ = cv2.Rodrigues(rvec)
    return np.hstack((rmtx,tvec))
global obj_points
global img_points

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
vertical, horizon = 7, 9  # target corners in vertical and horizontal direction

grid = np.zeros((vertical * horizon, 3), np.float32)
grid[:, :2] = np.mgrid[:horizon, :vertical].T.reshape(-1, 2)

obj_points = []  # 3d point in real world space
img_points = []  # 2d points in image plane

image_list = glob.glob(os.path.join('C:\\Users\\chuyangl\\Desktop\\liushuai\\calibrator\\board', "*.bmp"))
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
''''
# 63 points
ret, matrix, distortion, rotation, translation = \
    cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

print("projMtx:\n",getProjMtx(np.array(rotation[0]),np.array(translation[0])))
projMtx = getProjMtx(np.array(rotation[0]),np.array(translation[0]))
_3d = np.array([8,0,0,1])
_2d = np.dot(np.dot(matrix,projMtx,),_3d)
_2d = _2d/_2d[2]
print("2d:",_2d,img_points[0][7])

mean_error = 0

new_img_points, _ = cv2.projectPoints(obj_points[0], rotation[0], translation[0], matrix, distortion)#3D点投影到平面
error = cv2.norm(img_points[0], new_img_points, cv2.NORM_L2) 
mean_error += error
print("new and old:",new_img_points[1],img_points[0][1])
print("mean error: ", mean_error )

print('*' * 12)
randn_number = 4
randn_index = np.random.randint(0,63,randn_number)

obj_points_ = [obj_points[0][randn_index]]
img_points_ = [img_points[0][randn_index]]
ret, matrix, distortion, rotation, translation = \
    cv2.calibrateCamera(obj_points_ , img_points_, (640,480), matrix,None)

projMtx = getProjMtx(np.array(rotation[0]),np.array(translation[0]))
print("projMtx: ",projMtx)
#new_img_points2 = np.dot(np.dot(matrix,projMtx,),obj_points[0])
#print(new_img_points2.shape,img_points[0].shape)
#error = cv2.norm(img_points[0], new_img_points2, cv2.NORM_L2) 

_3d = np.array([8,0,0,1])
_2d = np.dot(np.dot(matrix,projMtx,),_3d)
_2d = _2d/_2d[2]
print("2d cord",_2d)
'''

def judge(number_points):
    '''
    judge the diff between less points and all points
    
    '''
    global obj_points
    global img_points
    # 63 points
    _3d = np.array([8,0,0,1])

    ret, matrix, distortion, rotation, translation = \
    cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    #print("projMtx 63 points:\n",getProjMtx(np.array(rotation[0]),np.array(translation[0])))
    projMtx = getProjMtx(np.array(rotation[0]),np.array(translation[0]))
    _3d = np.array([0,0,0,1])
    _2d = np.dot(np.dot(matrix,projMtx,),_3d)
    _2d = _2d/_2d[2]
    #print("2d:",_2d,img_points[0][7])

    mean_error = 0
    new_img_points, _ = cv2.projectPoints(obj_points[0], rotation[0], translation[0], matrix, distortion)#3D点投影到平面
    error = cv2.norm(img_points[0], new_img_points, cv2.NORM_L2) 
    mean_error += error
    #print("new and old:",new_img_points[1],img_points[0][1])
    #print("mean error: ", mean_error )

    print('*' * 12)
    randn_number = number_points
    randn_index = np.random.randint(0,63,randn_number)

    obj_points_ = [obj_points[0][randn_index]]
    img_points_ = [img_points[0][randn_index]]
    
    ret, matrix, distortion, rotation, translation = \
        cv2.calibrateCamera(obj_points_ , img_points_, (640,480), matrix,None)

    projMtx = getProjMtx(np.array(rotation[0]),np.array(translation[0]))
    #print("projMtx randn points: ",projMtx)
    #new_img_points2 = np.dot(np.dot(matrix,projMtx,),obj_points[0])
    #print(new_img_points2.shape,img_points[0].shape)
    #error = cv2.norm(img_points[0], new_img_points2, cv2.NORM_L2) 
    _2d_ = np.dot(np.dot(matrix,projMtx,),_3d)
    _2d_ = _2d_/_2d_[2]
    #print("2d cord",_2d)
    print("img points:",img_points[0][0])
    print("Points63 ---Points---",number_points,"\n",_2d,"\n",_2d_)
    print("distance: ",np.sqrt(np.sum(np.square(_2d - _2d_))))
for item in [4,6,8,10,12,14,16,63]:
    judge(item)

