import cv2
import numpy as np

#imgl = cv2.imread('left-remap.jpg')
#imgr = cv2.imread('right-remap.jpg')

imgl = cv2.imread('../stereo512/left.bmp')
imgr = cv2.imread('../stereo512/right.bmp')

image = np.hstack((imgl, imgr))
#image = np.concatenate((gray1, gray2)) #纵向连接=

for i in range(10):
    ptStart = (0, 48 * i)
    ptEnd = (1280, 48 * i)
    point_color = (0, 255, 0) # BGR
    thickness = 1 
    lineType = 4
    cv2.line(image, ptStart, ptEnd, point_color, thickness, lineType)


#cv2.imshow('img', image)

imgl = cv2.imread('left-remap.jpg')
imgr = cv2.imread('right-remap.jpg')
image = np.hstack((imgl, imgr))
#image = np.concatenate((gray1, gray2)) #纵向连接=

for i in range(10):
    ptStart = (0, 48 * i)
    ptEnd = (1280, 48 * i)
    point_color = (0, 255, 0) # BGR
    thickness = 1 
    lineType = 4
    cv2.line(image, ptStart, ptEnd, point_color, thickness, lineType)


cv2.imshow('img', image)

cv2.waitKey(0)