# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:27:11 2021

@author: imada
"""


#%% IMPORT LIBRARIES

import pandas as pd
from feature_scaling import feature_scaling
import numpy as np



#%% FEATURE ENGINEER SOME OF THE INPUT PARAMETERS
"""
-----------------------------------------------------------------

FEATURE ENGINEER SOME OF THE INPUT PARAMETERS

-----------------------------------------------------------------
"""

def feature_engineer_input(dataset, time_to_cyclic, do_feature_scaling, n_locations):        
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # ONE-HOT-ENCODING of categorical features
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # Get one hot encoding of columns "Context"
    FEATTMP = pd.get_dummies(dataset.Context, prefix='Context')
    # Drop column Context as it is now encoded
    dataset = dataset.drop('Context',axis = 1)
    # Join the encoded dataset
    dataset = dataset.join(FEATTMP)
        
    # Get one hot encoding of columns "Road_direction"
    FEATTMP = pd.get_dummies(dataset.Road_direction, prefix='Road_direction')
    # Drop column Context as it is now encoded
    dataset = dataset.drop('Road_direction',axis = 1)
    # Join the encoded dataset
    dataset = dataset.join(FEATTMP)
    
    
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    # COMPUTE LAGS Features
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
    # # 2-week lag per location (7 days per week and)
    # dataset['lag_2w'] = dataset['traffic_intensity_plus_60min'].shift(24*14*n_locations)
    # # 1-week lag per location (7 days per week and)
    # dataset['lag_1w'] = dataset['traffic_intensity_plus_60min'].shift(24*7*n_locations)
    # 2 hours lag per location
    dataset['lag_2h'] = dataset['traffic_intensity_plus_60min'].shift(2*n_locations)
    # 1 hour lag per location
    dataset['lag_1h'] = dataset['traffic_intensity_plus_60min'].shift(1*n_locations)
    # remove NaN
    dataset.dropna(inplace=True)
    
#    dataset['diff_w0'] = dataset['lag_2w'] - dataset['lag_1w']
    
    
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    # CONVERT TIME TO CYCLES
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

    # http://blog.davidkaleko.com/feature-engineering-cyclical-features.html
    # We map each cyclical variable onto a circle such that the lowest value for
    # that variable appears right next to the largest value.
    if time_to_cyclic:
        dataset['hours_sin'] = np.sin(dataset.hours*(2.*np.pi/24))
        dataset['hours_cos'] = np.cos(dataset.hours*(2.*np.pi/24))
        dataset['days_sin'] = np.sin((dataset.days-1)*(2.*np.pi/7))
        dataset['days_cos'] = np.cos((dataset.days-1)*(2.*np.pi/7))
        dataset['Weeks_sin'] = np.sin((dataset.Weeks-1)*(2.*np.pi/52))
        dataset['Weeks_cos'] = np.cos((dataset.Weeks-1)*(2.*np.pi/52))    
        dataset = dataset.drop(["hours","days","Weeks"],axis = 1)
                
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    # NORMALIZE & STANDARDIZE
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
    if do_feature_scaling:
        dataset[['precipitation_rate_mm','traffic_speed','lag_2h','lag_1h']]  = \
            feature_scaling(dataset[['precipitation_rate_mm','traffic_speed','lag_2h','lag_1h']].to_numpy()) 
        # dataset[['precipitation_rate_mm','traffic_speed','lag_1w','lag_2h','lag_1h']]  = \
        #     feature_scaling(dataset[['precipitation_rate_mm','traffic_speed','lag_1w','lag_2h','lag_1h']].to_numpy()) 
    
    
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    # RE-ARRANGE COLUMNS
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
    # dataset = dataset.drop(["X_ID","Y_ID"],axis = 1)
#    dataset = dataset.drop(["loc_ID"],axis = 1)
#    dataset = dataset.drop(["lag_2w"],axis = 1)
            
    ytmp = dataset.traffic_intensity_plus_60min
    # Drop column traffic_intensity
    dataset = dataset.drop('traffic_intensity_plus_60min',axis = 1)
    # Join it at end of dataset
    dataset = dataset.join(ytmp)
    
    # Input
    features_names = dataset.columns.values[0:-1]
    
    # Output: traffic_intensity_plus_60min estimate
    LABEL    = "traffic_intensity_plus_60min"

    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    # RETURN
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

    # Reset index and use the drop parameter to avoid the old index being added as a column
    dataset = dataset.reset_index(drop=True)

    return dataset, features_names, LABEL;

    
    
    