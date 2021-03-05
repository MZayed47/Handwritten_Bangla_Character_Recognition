# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 23:43:36 2020

@author: Mashrukh
"""

import cv2
import os

def loadImages(path='.'):
    
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.png')]
    '''
    all_files = os.listdir(path)    #Lists all files in the given directory
    #print(all_files)
    
    images=[]
    for i in all_files:
        if(i.endswith('png')):      #Seelcts only the images ending with .png
            images.append(i)
    
    print(images)                   #Prints all the image file names with .png
    '''


print(loadImages())                 #Prints all the values returned form the function

filenames =loadImages()

for file in filenames:
    img = cv2.imread(file,0)
    print(img)
    print()



'''
images=[]
for file in filenames:
    images.append(cv2.imread(file,cv2.IMREAD_UNCHANGED))
    
print(images)
'''


'''
num = 0
for image in images:
    cv2.imwrite(str(num)+'.png', image)
    num+=1
'''
