# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 22:29:18 2020

@author: KW0495
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import random as rd

#reading the input file
df = pd.read_excel ("datafile.xls")

#factors dictionary
factors={}
factors[0]='Salinity'
factors[1]='Dried up'
factors[2]='Destroyed beyond repair'
factors[3]='Sea water intrusion'
factors[4]='Industrial effluents'
factors[5]='Availability of Major Medium Irrigation Projects'
factors[6]='Other reasons'
factors[7]='total effect'

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

#creating new column to find total effect
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

#normalizing the columns
c_df=(c_df-c_df.min())/(c_df.max()-c_df.min())

for col in range(0,8):
    x = c_df.iloc[:, [2*col,2*col+ 1]].values
    
    m=x.shape[0] #number of records
    n=x.shape[1] #dimensionality number 
    n_iter=100 #number of iterations
    
    K=3 # number of clusters
    
    #assigning the centroids at random positions
    Centroids=np.array([]).reshape(n,0) 
    
    for i in range(K):
        rand=rd.randint(0,m-1)
        Centroids=np.c_[Centroids,x[rand]]
    
    #calculating the euclidian distance from centroids to each of the points and assign them to one cluster
    Output={}
    
    EuclidianDistance=np.array([]).reshape(m,0)
    for k in range(K):
           tempDist=np.sum((x-Centroids[:,k])**2,axis=1)
           EuclidianDistance=np.c_[EuclidianDistance,tempDist]
    C=np.argmin(EuclidianDistance,axis=1)+1
    
    #calculating the new centroids for the clusters
    Y={}
    for k in range(K):
        Y[k+1]=np.array([]).reshape(2,0)
    for i in range(m):
        Y[C[i]]=np.c_[Y[C[i]],x[i]]
         
    for k in range(K):
        Y[k+1]=Y[k+1].T
        
    for k in range(K):
         Centroids[:,k]=np.mean(Y[k+1],axis=0)
         
    #repeating the process for n iterations
    for i in range(n_iter):
          EuclidianDistance=np.array([]).reshape(m,0)
          for k in range(K):
              tempDist=np.sum((x-Centroids[:,k])**2,axis=1)
              EuclidianDistance=np.c_[EuclidianDistance,tempDist]
          C=np.argmin(EuclidianDistance,axis=1)+1
          Y={}
          for k in range(K):
              Y[k+1]=np.array([]).reshape(2,0)
          for i in range(m):
              Y[C[i]]=np.c_[Y[C[i]],x[i]]
         
          for k in range(K):
              Y[k+1]=Y[k+1].T
        
          for k in range(K):
              Centroids[:,k]=np.mean(Y[k+1],axis=0)
          Output=Y
    
    #plotting the graphs
    color=['red','blue','green']
    labels=['cluster1','cluster2','cluster3']
   
    fig, ax = plt.subplots()
    for k in range(K):
        ax.scatter(Output[k+1][:,0],Output[k+1][:,1],c=color[k],label=labels[k])
    ax.scatter(Centroids[0,:],Centroids[1,:],s=50,c='black',label='Centroids')
    plt.xlabel('Number of dugwells', fontsize=16)
    plt.ylabel('Irrigation Potential lost in hectares', fontsize=16)
    string='./plot/kmeans_'+factors[col]+'.jpg'
    fig.savefig(string)

