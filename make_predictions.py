# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:19:08 2021

@author: imada
"""

#%% IMPORT LIBRARIES

from import_dataset import import_dataset
from plot_map import make_plot_map
from save_predictions_to_json import save_predictions_to_json
from make_data_lag import series_to_supervised
import pandas as pd
import pickle
import glob
import numpy as np

#%%
saved_model_filename = './saved_models/RF_20210916115453317.sav'
saved_data_transformer = './saved_models/col_transform_20210916115453317.sav'

new_Xinput_filename_for_predictions = "./data/new_Xinput_for_predictions.csv"
file_path_predictions_to_json = './output_model_predictions.json'


#%%

# Import new input X data to make brand new predictions
new_Xinput_for_predictions = import_dataset(new_Xinput_filename_for_predictions)

# Introduce the traffic intensity lag time series as input features for supervised learning
dataset,n_locations = series_to_supervised(new_Xinput_for_predictions,kstepsahead=2)

new_X = dataset.iloc[:, :-1]

xcoord = np.array(new_X.X_ID[0:n_locations]); 
ycoord = np.array(new_X.Y_ID[0:n_locations]);  
locID  = np.array(new_X.loc_ID[0:n_locations]);  

# make new predictions
saved_model_filename=glob.glob(saved_model_filename)
saved_data_transformer=glob.glob(saved_data_transformer)

# load the model and data transformer from disk
model = pickle.load(open(saved_model_filename[0], 'rb'))
data_transformer = pickle.load(open(saved_data_transformer[0], 'rb'))

# transform new data with original data transformer
X_transformer = data_transformer.transform(new_X)

# Predict using loaded model
ypredict_from_saved_model = model.predict(X_transformer)    

# Save predictions to JSON file
save_predictions_to_json(file_path_predictions_to_json,ypredict_from_saved_model)    

# Visualize predictions on a map-like plot of road locations
xcoord = np.array([])
is_empty_xcoord = xcoord.size == 0
is_empty_ycoord = ycoord.size == 0

if is_empty_xcoord or is_empty_xcoord:
    print('ERROR: Coordinates not fully defined. Cannot plot results on map.')
else:
    make_plot_map(xcoord,ycoord,locID,ypredict_from_saved_model)    


