import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from albumentations import CLAHE


def calibrate(nx, ny):
    """
    Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    """
    objpoints = []
    imgpoints = []
    
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    img = mpimg.imread('../camera_cal/calibration2.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if ret == True:

        imgpoints.append(corners)
        objpoints.append(objp)
        resp, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        offset = 100
        image_size = (gray.shape[1], gray.shape[0])
        dst = np.float32([[offset, offset],
                        [image_size[0]-offset, offset],
                        [image_size[0]-offset, image_size[1]-offset],
                        [offset, image_size[1]-offset]])
        return mtx, dist



def IMG(image):
    img = mpimg.imread(image)
    img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
    width = img.shape[1]
    height = img.shape[0]
    return img, width, height

img, width, height = IMG('../data/image/GH040022/1589131910.8781476.jpg')


def display(img, modif, title):

    while True:
        
        cv2.imshow('origin', img)
        cv2.imshow(title, modif)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def undist(img):
    """
    Apply a distortion correction to raw images.
    """
    mtx, dist = calibrate(9, 6)
    undist_pic = cv2.undistort(img, mtx, dist, None, mtx)
    
    return undist_pic, img

undist_pic, img = undist(img)

display(img, undist_pic, 'undist')


def warp(img):
    print(img.shape)
    offset = 300
    src = np.float32(
            [
            [200, 150],
            [200, 330],
            [860, 330],
            [860, 150]
            ])
    print(src)
    dst = np.float32(
            [[(width / 4), 0],
            [(width / 4), height],
            [(width * 5 / 4), height],
            [(width * 5 / 4), 0]])
    print(dst)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_LINEAR)
    
    return warped


warp = warp(undist_pic)
display(img, warp, 'warp')

im = cv2.rectangle(undist_pic,(200, 150),(860,330),(0,255,0),3)
display(img, im, 'test')
