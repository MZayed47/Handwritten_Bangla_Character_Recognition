# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:53:07 2020

@author: Hp
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

print(x_train.shape)

new_x = x_train.values.reshape(8000, 28,28,1)

print(new_x.shape)
