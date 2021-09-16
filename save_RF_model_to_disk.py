# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 08:32:35 2021

@author: imada
"""


#%% IMPORT LIBRARIES

import pickle
import datetime


#%% FUNCTION TO SAVE RF MODEL TO DISK
"""
-----------------------------------------------------------------

FUNCTION TO SAVE RF MODEL TO DISK

-----------------------------------------------------------------
"""     
        
# Function to load and parse the json file
def save_RF_model_to_disk(RF_model,col_transform):
    
    model_id = '{}'.format(
    datetime.datetime.strftime(datetime.datetime.now(datetime.timezone.utc), "%Y%m%d%H%M%S%f")[:-3])
    
    saved_model_filename = './saved_models/RF_'+model_id+'.sav'
    saved_data_transformer = './saved_models/col_transform_'+model_id+'.sav'

    # save the model to disk with pickle
    pickle.dump(RF_model, open(saved_model_filename, 'wb'))
    
    # save the data ColumnTransformer to disk with pickle 
    pickle.dump(col_transform, open(saved_data_transformer, 'wb'))
   
    return saved_model_filename, saved_data_transformer