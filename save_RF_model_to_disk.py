# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 08:32:35 2021

@author: imada
"""


#%% IMPORT LIBRARIES

import pickle


#%% FUNCTION TO SAVE RF MODEL TO DISK
"""
-----------------------------------------------------------------

FUNCTION TO SAVE RF MODEL TO DISK

-----------------------------------------------------------------
"""     
        
# Function to load and parse the json file
def save_RF_model_to_disk(RF_model,filename_to_save):

    # save the model to disk with pickle
    pickle.dump(RF_model, open(filename_to_save, 'wb'))
    