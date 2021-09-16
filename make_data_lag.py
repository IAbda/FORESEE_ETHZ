# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 23:46:59 2021

@author: imada
"""




#%% IMPORT LIBRARIES

import numpy as np

#%% create data lag
"""
-----------------------------------------------------------------

CONVERT TIME SERIES INTO A SUPERVISED LEARNING PROBLEM
This allows us to split data in train/test sets in the usual way

-----------------------------------------------------------------
"""

def series_to_supervised(dset,kstepsahead=1,dropnan=True):         
    
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    # COMPUTE LAGS Features
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
        
    # First column in dset is always location ID
    # Get unique locations/nodes/intersections ID from first column of dset
    column_values_loc_ID = dset.iloc[:,0].values
    unique_column_values_loc_ID = np. unique(column_values_loc_ID)
    n_locations=len(unique_column_values_loc_ID)

    # Final column in dset is always the QoI, namely, traffic intensity
    traffic_intensity_id = dset.columns[-1]
    
    # Input historical sequence
    for i in range(kstepsahead,0,-1):
        dset["lag(t-%sh)"%(i)] = dset[traffic_intensity_id].shift(i*n_locations)

    # t+k steps ahead  sequence
    for i in range(0,kstepsahead):
        dset["lag(t+%sh)"%(i)] = dset[traffic_intensity_id].shift(-i*n_locations)
        
    # remove NaN after shift
    if dropnan:
        dset.dropna(inplace=True)
        
    # index_no = dset.columns.get_loc(traffic_intensity_id)    
    # ytmp = dset.traffic_intensity
    # Drop column traffic_intensity
    dset = dset.drop(traffic_intensity_id,axis = 1)
    # Join it at end of dset
    # dset = dset.join(ytmp)
    
    return dset,n_locations;

    
    
    