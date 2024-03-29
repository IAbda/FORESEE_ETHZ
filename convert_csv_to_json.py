# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 08:50:36 2021

@author: imada
"""


#%% IMPORT LIBRARIES

import csv 
import json 

#%% FUNCTION TO CONVERT CSV TO JSON 
"""
-----------------------------------------------------------------

FUNCTION TO CONVERT CSV TO JSON      

-----------------------------------------------------------------
"""     

# Convert some strings to numeric (as they are expected to be numeric)
# where one does not have to know a-priori which values are numeric
def numerify(row):
    for k, v in list(row.items()):
        try:
            row[k] = int(v)
        except ValueError:
            try:
                row[k] = float(v)
            except ValueError:
                pass

        
# Function to convert a CSV to JSON 
# Takes the file paths as arguments 
def make_json(file, json_file):
    csv_rows = []
    
    # Open a csv reader called DictReader     
    with open(file) as csvfile:
        reader_ind = csv.DictReader(csvfile)
        field = reader_ind.fieldnames

        # Convert each row into a dictionary  
        # and add it to csv_rows         
        for row in reader_ind:
            numerify(row)
            csv_rows.extend([{field[i]:row[field[i]] for i in range(len(field))}])
            
       # Open a json writer, and use the json.dumps()  
       # function to dump data     
        with open(json_file, "w") as f:
            f.write(json.dumps(csv_rows, sort_keys=False, indent=4, separators=(',', ': '))) #for pretty
            #f.write(json.dumps(csv_rows))


#%% Driver Code 

# # Decide the two file paths according to your  
# # computer system 
# csvFilePath = "./Data/OutGenTrafficSyntheticSamples.csv"
# jsonFilePath = "./Data/OutGenTrafficSyntheticSamples.json"
  
# # Call the make_json function 
# make_json(csvFilePath, jsonFilePath)

