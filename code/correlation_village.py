#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 20:48:34 2020

@author: gsk98
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
from numpy import cov

#reading the data
df = pd.read_excel ('datafile.xls')
df.drop(["District", "Block/Tehsil","State"], axis = 1, inplace = True) 


#summing up all the data belonging to a single village
c_df=df.groupby(['Village']).sum()

#finding correlation among different factors
cor_matrix = np.zeros((7,7), dtype='float64')
col_dict={}

col_dict[0]='Salinity - No.'
col_dict[1]='Dried up - No.'
col_dict[2]='Destroyed beyond repair - No.'
col_dict[3]='Sea water intrusion  - No.'
col_dict[4]='Industrial effluents - No.'
col_dict[5]='Availability of Major/Medium Irrigation Projects  - No.'
col_dict[6]='Other reasons - No.'


for i in range(0,7):
    for j in range (0,7):
        cov_matrix=cov(c_df[col_dict[i]],c_df[col_dict[j]])
        covariance=cov_matrix[0][1]
        cor_matrix[i][j]=((covariance)/((std(c_df[col_dict[i]]))*(std(c_df[col_dict[j]]))))


