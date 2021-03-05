# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 10:34:45 2020

@author: Mashrukh
"""

import pandas as pd

# Importing data & selecting the rows

mnist_data = pd.read_csv('mnist_test.csv')
#data.isnull().any()
#data.drop("3P%", axis=1, inplace=True)

x = mnist_data.iloc[:,1:785]
y = mnist_data.iloc[:,0]



# Splitting data into test set & train set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)



# Feature scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_trainN = sc.fit_transform(x_train)
x_testN = sc.transform(x_test)



# Change the labels from categorical to one-hot encoding

from keras.utils import to_categorical

y_trainN = to_categorical(y_train)
y_testN = to_categorical(y_test)


# ANN

import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

bclfr = Sequential()

bclfr.add(Dense(units=100, kernel_initializer='uniform', activation='relu', input_dim=784))

bclfr.add(Dense(units=100, kernel_initializer='uniform', activation='relu'))

bclfr.add(Dense(units=10, kernel_initializer='uniform', activation='sigmoid'))

bclfr.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



# Train & Testing

bclfr.fit(x_trainN, y_trainN, batch_size=10, epochs=100)

y_pred = bclfr.predict(x_testN)









