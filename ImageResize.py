# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 09:46:48 2020

@author: Hp
"""

import cv2

img = cv2.imread('Grasshopper.jpg')

scaler = 0.5

width = int(img.shape[1]*scaler)
height = int(img.shape[0]*scaler)

dimension = (width,height)

print(dimension)

img_resized = cv2.resize(img, dimension)

print(img_resized.shape)

cv2.imshow('Resized Image', img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
