# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:11:44 2021

@author: imada
"""


#%% IMPORT LIBRARIES

import pandas as pd


#%% IMPORT THE DATA

"""
-----------------------------------------------------------------

IMPORT DATA FROM CSV FILE 

-----------------------------------------------------------------
"""

def import_dataset(filename):
    # Import the CSV data file with Panda
    dataset = pd.read_csv(filename)
    
    return dataset

