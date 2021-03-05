# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:36:53 2020

@author: Mashrukh
"""

import csv
import pandas as pd

f = open('mnist_test.csv')
csv_f = csv.reader(f)
mnist = pd.read_csv('mnist_test.csv')

x=1

for row in csv_f:           #Reads values row by row
    print(row)              #Prints all the values row by row
    #print(row[0])           #Prints only first value of each row
    
    '''
    for i in row:
        print(i)            #Prints each value of a specific row
    '''
    
    x=x+1
    if(x>2):
        break
