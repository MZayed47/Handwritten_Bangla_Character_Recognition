# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:07:03 2020

@author: Hp
"""

import cv2
import os


all_files = os.listdir(path='.')    #Lists all files in the given directory
#print(all_files)

dimension = (28,28)
num=0

for i in all_files:
    if(i.endswith('png')):      #Seelcts only the images ending with .png
        img = cv2.imread(i)
        img_resized = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
        #cv2.imshow('Resized Image', img_resized)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        cv2.imwrite(str(num)+'.png', img_resized)
        num+=1
