# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 23:11:52 2020

@author: KW0495
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
def DBSCAN(x, eps, MinPts):

    
    #This variable has the final list of assigned labels.Initially they're all 
    #set to zero. If there are no neighbors the point will be assigned as -1.
        
    answer_labels = [0]*len(x)

    # current_itr is the label that will be assigned to the new cluster   
    current_itr = 0
    
   #This loop is to assign each of the points a particular label
    for k in range(0, len(x)):
    
        #The points that are not already assigned are considered
        if not (answer_labels[k] == 0):
           continue
        
        # This function returns all the neighbors of this particular point
        n_array = neighborQuery(x, k, eps)
        
        #if the minimum number of points are less than our threshold we 
        #consider the point as outlier
        if len(n_array) < MinPts:
            answer_labels[k] = -1
        # Else we consider the cluster and assign the label to all it's
        # neighbors as well
        else: 
           current_itr += 1
           expandCluster(x, answer_labels, k, n_array, current_itr, eps, MinPts)
    
    # Returning the final set of labels
    return answer_labels


def expandCluster(x, answer_labels, k, n_array, current_itr, eps, MinPts):
   
    # Assigning the cluster to the point
    answer_labels[k] = current_itr
    
    #Assigning the cluster label to all it's neighbors 
    i = 0
    while i < len(n_array):    
        
        point = n_array[i]
       
        #if the point was labeled as noise we label it to the current_itr
        if answer_labels[point] == -1:
           answer_labels[point] = current_itr
        
        # else if the point is not labelled we label it to the current_itr
        # and all it's neighbors too are added in the array because they're branch points
        elif answer_labels[point] == 0:
            
            answer_labels[point] = current_itr
            
            # Find all the neighbors of point
            relative_n_array = neighborQuery(x, point, eps)
            
            
            if len(relative_n_array) >= MinPts:
                n_array = n_array + relative_n_array
           
        i += 1        
    
    


def neighborQuery(x, k, eps):

    neighbors = []
    
    # For each point in the dataset...
    for point in range(0, len(x)):
        
        # If the distance is below the threshold, add it to the neighbors list.
        if np.linalg.norm(x[k] - x[point]) < eps:
           neighbors.append(point)
            
    return neighbors



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
    
    x = c_df.iloc[:, [2*col,2*col+1]].values 
    #getting the label values
    my_labels = DBSCAN(x, eps=0.3, MinPts=2)
    
    #plotting and saving the figures
    fig, ax = plt.subplots()
    colors = ['red','green','blue','black']
    color= ['red' if l == 1  else 'green' for l in my_labels]
    ax.scatter(x[:,0],x[:,1],c=my_labels, cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlabel('Number of dugwells', fontsize=16)
    plt.ylabel('Irrigation Potential lost in hectares', fontsize=16)
    string='./plot/dbscan_'+factors[col]+'.jpg'
    fig.savefig(string)



