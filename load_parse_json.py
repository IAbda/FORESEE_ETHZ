# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 08:50:36 2021

@author: imada
"""


#%% IMPORT LIBRARIES

import json 
import pandas as pd


#%% FUNCTION TO LOAD AND PARSE THE JSON FILE 
"""
-----------------------------------------------------------------

FUNCTION TO LOAD AND PARSE THE JSON FILE

-----------------------------------------------------------------
"""     
        
# Function to load and parse the json file
def load_parse_json(jsonFilePath):

    with open(jsonFilePath) as f:
        data = json.load(f)        
    # Convert dict to dataframe    
    dataset = pd.DataFrame.from_dict(data)    
    # print(dataset)

    return dataset
    
