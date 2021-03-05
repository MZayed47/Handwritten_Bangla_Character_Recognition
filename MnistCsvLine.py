# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:31:00 2020

@author: Mashrukh
"""

import csv

with open("mnist_test.csv", "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        print('line[{}] = {}'.format(i, line))
        #print(line)
