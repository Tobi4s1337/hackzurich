# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 11:35:58 2020

@author: dkloe
"""

import numpy as np
import cv2
import sys
from PIL import Image

font = cv2.FONT_HERSHEY_COMPLEX
# -- coding: utf-8 --
src = cv2.imread('sketch4.jpg')
src = cv2.fastNlMeansDenoising(src,None,10,7,21)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

src = image_resize(src, width = 640)

width = 640
height = src.shape[0]
dsize = (width, height)
src = cv2.resize(src, dsize)
org = src.copy()
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(gray,(9,9),0)
th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY_INV,11,2)
cv2.imshow('ImageWindow', th)
cv2.waitKey()
th2 = th.copy()
print(th)
bw  = cv2.ximgproc.thinning(th)
cv2.imwrite("skel.pgm",bw)
cv2.imwrite("th.pgm",th2)

## detection of ground, capacitor, v_source

cv2.imshow('ImageWindow', bw)
cv2.waitKey()

kernel = np.ones((1,5), np.uint8)  # note this is a horizontal kernel
d_im = cv2.dilate(bw, kernel, iterations=1)
e_im = cv2.erode(d_im, kernel, iterations=1) 


_, threshold = cv2.threshold(e_im, 245, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(e_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
counter = 1
for cnt in contours:
    
    approx = cv2.approxPolyDP(cnt, 0.001*cv2.arcLength(cnt, True), True)
    cv2.drawContours(src, [approx], 0, (255,0,0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1]

    

    if 50 < len(cnt) < 200:
        
        print(counter)
        print("Len CNT:", len(cnt))
        print("Len approx:", len(approx))
        
        
        
        
        if len(approx) == 3:
            cv2.putText(src, "Triangle", (x, y), font, 1, (0))
        elif len(approx) == 4:
            cv2.putText(src, "Rectangle", (x, y), font, 1, (0))
        elif len(approx) == 5:
            cv2.putText(src, "Pentagon", (x, y), font, 1, (0))
        elif 6 < len(approx) < 15:
            cv2.putText(src, "Ellipse", (x, y), font, 1, (0))
        elif 20 < len(approx) < 150:
            cv2.putText(src,"Circle", (x, y), font, 1, (0))

        counter = counter + 1

cv2.imshow('ImageWindow', e_im)
cv2.imshow("src.jpg", src) 
cv2.waitKey()
color = cv2.cvtColor(e_im,cv2.COLOR_GRAY2RGB)
