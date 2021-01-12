# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 08:50:36 2021

@author: imada
"""


#%% IMPORT LIBRARIES

import matplotlib.pyplot as plt
 

#%% FUNCTION TO CONVERT CSV TO JSON 
"""
-----------------------------------------------------------------

FUNCTION TO PLOT MAP      

-----------------------------------------------------------------
"""     
        
# Function to convert a CSV to JSON 
# Takes the file paths as arguments 
def make_plot_map(dataset,n_locations):
    xcoord = dataset.X_ID[0:n_locations]; 
    ycoord = dataset.Y_ID[0:n_locations]; 
    locID  = dataset.loc_ID[0:n_locations];
    s = 50
    a = 0.4
    fig, ax = plt.subplots()
    ax.scatter(xcoord, ycoord, edgecolor='k',
                c="r", s=s, marker="o", alpha=a)
    plt.xlabel("Xcoord")
    plt.ylabel("Ycoord")
    for i, txt in enumerate(locID):
        ax.annotate(txt, (xcoord[i]+0.001, ycoord[i]+0.002))
        



#%% Driver Code 
"""  
# Decide the two file paths according to your  
# computer system 
csvFilePath = "./Data/OutGenTrafficSyntheticSamples.csv"
jsonFilePath = "./Data/OutGenTrafficSyntheticSamples.json"
  
# Call the make_json function 
make_json(csvFilePath, jsonFilePath)
"""
