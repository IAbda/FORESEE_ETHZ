# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 11:47:58 2021

@author: imada
"""



#%% IMPORT LIBRARIES
import json 


#%% FUNCTION TO SAVE MODEL OUTPUT PREDICTIONS TO JSON FILE   
"""
-----------------------------------------------------------------

FUNCTION TO SAVE MODEL OUTPUT PREDICTIONS TO JSON FILE   

-----------------------------------------------------------------
"""     
        
# Function to save model predictions to JSON 
# Takes the file paths as arguments 
def save_predictions_to_json(file_path_predictions_to_json,ypredict_from_saved_model):
    locID  = range(0,len(ypredict_from_saved_model));    
    zipbObj = zip(locID,ypredict_from_saved_model)
    aDict = dict(zipbObj)
    jsonString = json.dumps(aDict,indent=2) # indent=2 to dump each dictionary entry on a new line in JSON
    jsonFile = open(file_path_predictions_to_json, "w")
    jsonFile.write(jsonString)
    jsonFile.close()       
