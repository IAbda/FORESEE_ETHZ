# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:19:08 2021

@author: imada
"""

#%% IMPORT LIBRARIES

from import_dataset import import_dataset
from plot_map import make_plot_map
from predict_from_saved_RF_model import predict_from_saved_RF_model
from feature_engineer_input import feature_engineer_input
from initialize_vars import initialize_vars


#%% initialize internal variables
do_feature_scaling, time_to_cyclic, n_splits, n_locations = initialize_vars()


#%%
saved_model_filename = './saved_models/saved_RF_model.sav'
new_Xinput_filename_for_predictions = "./data/new_Xinput_for_predictions.csv"


#%%

# Import new input X data to make brand new predictions
new_Xinput_for_predictions = import_dataset(new_Xinput_filename_for_predictions)

# Feature engineer raw input features to make them ready for predictions
new_Xinput_for_predictions, features_names, LABEL = feature_engineer_input(new_Xinput_for_predictions, time_to_cyclic, do_feature_scaling, n_locations)

# make new predictions
ypredict_from_saved_model = predict_from_saved_RF_model(saved_model_filename,new_Xinput_for_predictions)
print(ypredict_from_saved_model)

# Visualize predictions on a map-like plot of road locations
make_plot_map(new_Xinput_for_predictions,ypredict_from_saved_model,n_locations)    

