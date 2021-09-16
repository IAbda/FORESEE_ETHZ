# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 10:04:28 2021

@author: imada
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 11:47:58 2021

@author: imada
"""



#%% IMPORT LIBRARIES
import json 
import numpy as np

#%% FUNCTION TO SAVE MODEL OUTPUT PREDICTIONS TO JSON FILE   
"""
-----------------------------------------------------------------

FUNCTION TO SAVE MODEL OUTPUT PREDICTIONS TO JSON FILE   

-----------------------------------------------------------------
"""     
        
# Function to save model predictions to JSON 
# Takes the file paths as arguments 
def save_model_performance_to_json(train_score, test_score, sorted_features_names, 
                                   feature_importances, saved_model_filename, saved_data_transformer, outputFile):
    
    # Write the output file
    Output = {
        "Model_ID": {
            "Trained_Model": saved_model_filename,
            "fitted_data_transformer": saved_data_transformer,
            },
        "Model_performance": {
            "train_score": train_score,
            "test_score": test_score,
            },
        "Model_permutation_feature_importance_test": {
            "feature_names":np.array(sorted_features_names).tolist(),
            "feature_importance":np.array(feature_importances).tolist() 
            },        
    }
    with open(outputFile, 'w') as f:
        json.dump(Output, f)
                   
