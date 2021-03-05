# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:20:41 2020

@author: Mashrukh
"""

import pandas

df = pandas.read_csv("mnist_test.csv")

#print(df.head(3))           # prints first 3 rows along with header & label
#print(df.loc[0,:])          # prints all the values of a specified column
#print(df.loc[0,1])          # prints all the values of a specified column

i=1
for i in df:
    print(df.loc[0,i])      #prints a specific row

