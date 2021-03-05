# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:36:20 2020

@author: Hp
"""

import pandas as pd

# Importing data & selecting the rows

mnist_data = pd.read_csv('mnist_train.csv')
#data.isnull().any()
#data.drop("3P%", axis=1, inplace=True)

x = mnist_data.iloc[:,1:785]
y = mnist_data.iloc[:,0]



# Splitting data into test set & train set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)



x_train = x_train.values.reshape(48000,28,28,1)
x_test = x_test.values.reshape(12000,28,28,1)

print(x_train.shape, x_test.shape)


# Feature scaling

x_trainN = x_train.astype('float32')
x_testN = x_test.astype('float32')
x_trainN = x_trainN / 255.
x_testN = x_testN / 255.



# Change the labels from categorical to one-hot encoding

from keras.utils import to_categorical

y_trainN = to_categorical(y_train)
y_testN = to_categorical(y_test)


# validation set

from sklearn.model_selection import train_test_split
x_trainN, x_valid, train_label, valid_label = train_test_split(x_trainN, y_trainN, test_size=0.2, random_state=13)

print(x_trainN.shape, x_valid.shape, train_label.shape, valid_label.shape)


# CNN

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

batch_size = 128
epochs = 20
num_classes = 10

fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1), padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(Dense(num_classes, activation='softmax'))


# Train & Testing

fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

fashion_model.summary()

fashion_train = fashion_model.fit(x_trainN, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_valid, valid_label))

test_eval = fashion_model.evaluate(x_testN, y_testN, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])


