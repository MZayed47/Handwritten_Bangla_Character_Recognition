# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:05:03 2020

@author: Hp
"""

import csv
import cv2
import os


#img = cv2.imread('aaa.png',0)    #the parameter 0 means greyscale
#print(img)

all_files = os.listdir(path='.')    #Lists all files in the given directory
#print(all_files)

for i in all_files:
    if(i.endswith('png')):      #Seelcts only the images ending with .png
        img = cv2.imread(i,0)
        img_array = img.reshape(1,784)
        
        with open('MyCsv60.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            
            for j in img_array:
                writer.writerow(j)





