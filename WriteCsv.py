# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 08:57:45 2020

@author: Mashrukh
"""

import csv

with open('MyCsv.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    
    writer.writerow(['label', '1*1', '1*2'])
    
    for i in range(1,10):
        writer.writerow(['1', '2', '3'])