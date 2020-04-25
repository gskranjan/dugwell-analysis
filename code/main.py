#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 11:55:59 2020

@author: gsk98
"""
#importing standard python packages
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


#reading the input file
df = pd.read_excel ('datafile.xls')


#dropping the columns that we wont be using for pre-processing
df.drop(["District", "Block/Tehsil","Village"], axis = 1, inplace = True) 

#storing all the states
states=df['State'].unique()
states=states.reshape(states.size,1)
states=np.concatenate(states).astype(str)

#Creating the dictionary to map states to a number 
state_dict={}
itr=0
for x in np.nditer(states):
    state_dict[str(x)]=itr
    state_dict[itr]=str(x)
    itr=itr+1

#label encoding the the states
for index, row in df.iterrows():
    df.at[index,'State']=state_dict[row['State']]
    
    
#grouping all the states and summing their rows
c_df=df.groupby(['State']).sum()

'''
plotting the total number of dugwells which are not being used in each state along with 
irrigation potential lost for each factor.
'''
itr=0
x=[]
y=[]
for (columnName, columnData) in c_df.iteritems():
    itr=itr+1
    if itr==1:
        x=columnData.values
    if itr==2:
        y=columnData.values
        name=""
        if columnName=="Availability of Major/Medium Irrigation Projects  - PL":
            name="Major medium irrigation projects"
        else:
            name=columnName.split("-")[0]
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        for i in range(0,23):
            ax.annotate(state_dict[i], (x[i], y[i])) 
        plt.xlabel('Number of dugwells', fontsize=16)
        plt.ylabel('Irrigation Potential lost in hectares', fontsize=16)
        fig.savefig('./plot/'+name+'.jpg')
        itr=0


'''
plotting the total number of dugwells which are not being used in each state along with 
irrigation potential lost for all of the factors.
'''
itr=0
c_df['total_no.']=0
c_df['total_pl']=0
for (columnName, columnData) in c_df.iteritems():
    itr=itr+1
    if itr==1:
        c_df['total_no.']=c_df['total_no.']+c_df[columnName]
    if itr==2:
        c_df['total_pl']=c_df['total_pl']+c_df[columnName]
        itr=0

fig, ax = plt.subplots()
ax.scatter(x, y)
for i in range(0,23):
    ax.annotate(state_dict[i], (x[i], y[i]))
plt.xlabel('Total Number of dugwells', fontsize=16)
plt.ylabel('Total Irrigation Potential lost in hectares', fontsize=16)
fig.savefig('./plot/Total_effect.jpg')


