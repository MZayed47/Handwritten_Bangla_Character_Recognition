# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 23:42:04 2020

@author: Mashrukh
"""

import pandas as pd
import numpy as np
import random


import keras
from keras.preprocessing import sequence
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Embedding, Flatten
from keras.layers import LSTM
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import imdb


# Importing data & selecting the rows

bangla_data = pd.read_csv('BanglaLekha_Vowel.csv')
#print(mnist_data.isnull().any())

x = bangla_data.iloc[:,1:785]
y = bangla_data.iloc[:,0]

# Splitting data into test set & train set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=int(random.uniform(0,100)))

'''
# Array Reshaping

x_train = x_train.values.reshape(x_train.shape[0],28,28,1)
x_test = x_test.values.reshape(x_test.shape[0],28,28,1)

print('Reshaped train arrray : ', x_train.shape, y_train.shape)
print('Reshaped test arrray : ', x_test.shape, y_test.shape)
'''


# Find the unique numbers from the train labels

classes = np.unique(y_train)
nClasses = len(classes)
print('Total number of classes : ', nClasses)
print('Output classes : ', classes)



# Feature scaling

x_trainN = x_train.astype('float32')
x_testN = x_test.astype('float32')
x_trainN = x_trainN / 255.
x_testN = x_testN / 255.



# Change the labels from categorical to one-hot encoding

from keras.utils import to_categorical

y_trainN = to_categorical(y_train)
y_testN = to_categorical(y_test)

print('Categorical y_train arrray : ', y_trainN.shape)



# Validation set

from sklearn.model_selection import train_test_split

x_trainN, x_valid, train_label, valid_label = train_test_split(x_trainN, y_trainN, test_size=0.2,
                                                               random_state=int(random.uniform(0,100)))

print('Final train shape : ', x_trainN.shape, train_label.shape)
print('Final validation shape : ', x_valid.shape, valid_label.shape)



'''
# CNN LSTM Model

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',
                            input_shape=(28,28,1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Dropout(0.25))


model = Sequential()
model.add(TimeDistributed(model, input_shape=(13940, 784, 1), dropout = 0.25, recurrent_dropout = 0.25)) 
model.add(LSTM(64, dropout = 0.25, recurrent_dropout = 0.25)) 
model.add(Dense(128, activation = 'sigmoid'))
model.add(Dense(11, activation='softmax'))
'''


#n_timesteps, n_features, n_outputs = x_trainN[1].shape, x_trainN[2].shape, y_trainN[1].shape

model = Sequential()
model.add(LSTM(128, dropout = 0.2, recurrent_dropout = 0.2))
model.add(Dense(11, activation='softmax'))
	

model.compile(loss=keras.losses.categorical_crossentropy,
                     optimizer=keras.optimizers.Adam(),metrics=['accuracy'])


model.fit(
   x_trainN, train_label,
   batch_size = 32,
   epochs = 5,
   verbose=1,
   validation_data = (x_valid, valid_label)
)

model.summary()

score, accuracy = model.evaluate(x_testN, y_testN, verbose=0) 
   
print('Test score:', score) 
print('Test accuracy:', accuracy)


