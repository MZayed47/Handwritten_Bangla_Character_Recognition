# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:08:53 2020

@author: Mashrukh
"""

from PIL import Image
import numpy as np
import cv2

img = Image.open('00081.png')
img_array = np.array(img)
#print(img_array)

img2 = cv2.imread('Grasshopper.jpg',0)    #the parameter 0 means greyscale
print(img2)

cv2.imshow('First Image', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imwrite('GreyHopper.jpg', img2)