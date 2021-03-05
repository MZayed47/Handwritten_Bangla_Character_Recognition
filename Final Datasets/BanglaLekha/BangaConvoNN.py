# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:41:09 2020

@author: Mashrukh
"""

import pandas as pd
import numpy as np
import random

# Importing data & selecting the rows

bangla_data = pd.read_csv('BanglaLekha_Vowel.csv')
#print(mnist_data.isnull().any())

x = bangla_data.iloc[:,1:785]
y = bangla_data.iloc[:,0]

print('Pixel Data : ', x.shape)
print('Label Data : ', y.shape)

# Splitting data into test set & train set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=int(random.uniform(0,100)))

print('Train dataset shape : ', x_train.shape, y_train.shape)
print('Test dataset shape : ', x_test.shape, y_test.shape)



# Display Train & Test Samples

import matplotlib.pyplot as plt


x_trainD = x_train.values.reshape(x_train.shape[0],28,28)
x_testD = x_test.values.reshape(x_test.shape[0],28,28)

plt.figure(1,figsize=[8,8])

# Display the first image in training data
plt.subplot(121)
plt.imshow(x_trainD[0,:,:], cmap='gray')
plt.title("Class : {}".format(y_train.iloc[0]))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(x_testD[0,:,:], cmap='gray')
plt.title("Class : {}".format(y_test.iloc[0]))



# Array Reshaping

x_train = x_train.values.reshape(x_train.shape[0],28,28,1)
x_test = x_test.values.reshape(x_test.shape[0],28,28,1)

print('Reshaped train arrray : ', x_train.shape, y_train.shape)
print('Reshaped test arrray : ', x_test.shape, y_test.shape)



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



# CNN

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

batch_size = 64
epochs = 5
num_classes = 11

bangla_model = Sequential()
bangla_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',
                            input_shape=(28,28,1), padding='same'))
bangla_model.add(LeakyReLU(alpha=0.1))
bangla_model.add(MaxPooling2D((2, 2),padding='same'))
bangla_model.add(Dropout(0.25))

bangla_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
bangla_model.add(LeakyReLU(alpha=0.1))
bangla_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
bangla_model.add(Dropout(0.25))

bangla_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
bangla_model.add(LeakyReLU(alpha=0.1))                  
bangla_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
bangla_model.add(Dropout(0.25))
bangla_model.add(Flatten())

bangla_model.add(Dense(128, activation='linear'))
bangla_model.add(LeakyReLU(alpha=0.1))
bangla_model.add(Dropout(0.3))

bangla_model.add(Dense(num_classes, activation='softmax'))



# Train & Testing

bangla_model.compile(loss=keras.losses.categorical_crossentropy,
                     optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

bangla_model.summary()

bangla_train = bangla_model.fit(x_trainN, train_label, batch_size=batch_size,
                    epochs=epochs,verbose=1,validation_data=(x_valid, valid_label))

test_eval = bangla_model.evaluate(x_testN, y_testN, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', 100*test_eval[1])



# Graphs

accuracy = bangla_train.history['accuracy']
val_accuracy = bangla_train.history['val_accuracy']
loss = bangla_train.history['loss']
val_loss = bangla_train.history['val_loss']
epochs = range(len(accuracy))


plt.figure(2,figsize=[8,6])
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')


plt.figure(3,figsize=[8,6])
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs')
plt.ylabel('Loss')



# Prdicted Labels

y_trainL = pd.Series(y_train).array
y_testL = pd.Series(y_test).array

predicted_classes = bangla_model.predict(x_testN)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
print(predicted_classes.shape, y_testL.shape)


correct = np.where(predicted_classes==y_testL)[0]
print("Found %d correct labels" % len(correct))

for i, correct in enumerate(correct[:9]):
    plt.figure(4,figsize=[9,9])
    plt.subplot(3,3,i+1)
    plt.imshow(x_testN[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_testL[correct]))
    plt.tight_layout()


incorrect = np.where(predicted_classes!=y_testL)[0]
print("Found %d incorrect labels" % len(incorrect))

for i, incorrect in enumerate(incorrect[:9]):
    plt.figure(5,figsize=[9,9])
    plt.subplot(3,3,i+1)
    plt.imshow(x_testN[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_testL[incorrect]))
    plt.tight_layout()



