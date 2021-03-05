# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 11:26:40 2020

@author: HP
"""

import PIL 
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.imagenet_utils import decode_predictions 
import matplotlib.pyplot as plt 
import numpy as np 
from keras.applications.resnet50 import ResNet50
from keras.applications import resnet50



filename = 'MashWhite.jpg'
## load an image in PIL format
original = load_img(filename, target_size = (224, 224))
print('PIL image size',original.size)
plt.imshow(original) 
plt.show()

#convert the PIL image to a numpy array 
numpy_image = img_to_array(original)

plt.imshow(np.uint8(numpy_image))

print('numpy array size',numpy_image.shape)

# Convert the image / images into batch format
image_batch = np.expand_dims(numpy_image, axis = 0)

print('image batch size', image_batch.shape)


# prepare the image for the resnet50 model
processed_image = resnet50.preprocess_input(image_batch.copy()) 

# create resnet model 
resnet_model = resnet50.ResNet50(weights = 'imagenet') 

# get the predicted probabilities for each class 
predictions = resnet_model.predict(processed_image)

# convert the probabilities to class labels 
label = decode_predictions(predictions) 

print(label)

