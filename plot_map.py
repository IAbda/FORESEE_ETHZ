# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 08:50:36 2021

@author: imada
"""


#%% IMPORT LIBRARIES

import matplotlib.pyplot as plt
import pandas as pd


#%% FUNCTION TO CONVERT CSV TO JSON 
"""
-----------------------------------------------------------------

FUNCTION TO PLOT MAP      

-----------------------------------------------------------------
"""     
        
# Function to convert a CSV to JSON 
# Takes the file paths as arguments 
def make_plot_map(xcoord,ycoord,locID,ypredict_from_saved_model):
    a = 0.5
    color = [float(i) for i in [str(item/255.) for item in ypredict_from_saved_model]]
    s = [25+2**n for n in [float(i) for i in [str(item/220.) for item in ypredict_from_saved_model]]]    
    fig, ax = plt.subplots()
    ax.scatter(xcoord, ycoord, edgecolor='k',
                c=color, s=s, marker="o", alpha=a)
    plt.xlabel("Road Xcoord")
    plt.ylabel("Road Ycoord")
    for i, txt in enumerate(locID):
        ax.annotate(txt, (xcoord[i]+0.001, ycoord[i]+0.002))
    for i, yvalpre in enumerate(ypredict_from_saved_model):
        ax.annotate('{:.0f}'.format(yvalpre), (xcoord[i]-0.002, ycoord[i]-0.0075))
    # set axes range
    # plt.xlim(-2, 2)
    plt.ylim(-3.76, -3.64)        



#%% Driver Code 
"""  
# Decide the two file paths according to your  
# computer system 
csvFilePath = "./Data/OutGenTrafficSyntheticSamples.csv"
jsonFilePath = "./Data/OutGenTrafficSyntheticSamples.json"
  
# Call the make_json function 
make_json(csvFilePath, jsonFilePath)
"""
