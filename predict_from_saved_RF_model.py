# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 08:29:59 2021

@author: imada
"""



#%% IMPORT LIBRARIES

import pickle



#%% FUNCTION TO LOAD SAVED RF MODEL AND MAKE PREDICTION
"""
-----------------------------------------------------------------

FUNCTION TO LOAD SAVED RF MODEL AND MAKE PREDICTION

-----------------------------------------------------------------
"""     
        
# Function to load and parse the json file
def predict_from_saved_RF_model(saved_model_filename,X_test):

    # load the model from disk
    loaded_model = pickle.load(open(saved_model_filename, 'rb'))

    # Predict using loaded model
    ypredict_from_saved_model = loaded_model.predict(X_test)
    
    return ypredict_from_saved_model
    