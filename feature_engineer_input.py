# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:27:11 2021

@author: imada
"""


#%% IMPORT LIBRARIES

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

#%% FEATURE ENGINEER SOME OF THE INPUT PARAMETERS
"""
-----------------------------------------------------------------

FEATURE ENGINEER SOME OF THE INPUT PARAMETERS

-----------------------------------------------------------------
"""

def feature_engineer_input(X,*Xopt):         
    

    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # ONE-HOT-ENCODING of categorical features and scaling
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # determine categorical and numerical features
    # define the data preparation for the columns

    numerical_ix = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_ix = X.select_dtypes(include=['object', 'bool']).columns
    
    t = [('cat', OneHotEncoder(handle_unknown = 'ignore'), categorical_ix)]
    # t = [('cat', OneHotEncoder(), categorical_ix), ('num', MinMaxScaler(), numerical_ix)]
    col_transform = ColumnTransformer(transformers=t, remainder='passthrough')
    col_transform.fit(X)
    Xout = col_transform.transform(X)    
    Xout = pd.DataFrame(Xout)
 
    Xout2 = pd.DataFrame()
    if not Xopt == False: # If tuple is not empty
        for arg in Xopt:
            Xout2 = col_transform.transform(arg) # apply same ColumnTransformer fitted on original X input data
            Xout2 = pd.DataFrame(Xout2)
    
    # Input features names
    features_names = np.array(col_transform.get_feature_names())     
    
    # Output label: traffic_intensity
    LABEL    = "traffic_intensity"
    
    # Normalize data
    """
    numerical_ix = Xout.select_dtypes(include=['int64', 'float64']).columns
    t = [('num', MinMaxScaler(), numerical_ix)]
    col_transform = ColumnTransformer(transformers=t, remainder='passthrough')
    # Xout= np.array(col_transform.fit_transform(Xout))
    Xout= col_transform.fit_transform(Xout)
    Xout = pd.DataFrame(Xout)
    """
            
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    # CONVERT TIME TO CYCLES
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

    # http://blog.davidkaleko.com/feature-engineering-cyclical-features.html
    # We map each cyclical variable onto a circle such that the lowest value for
    # that variable appears right next to the largest value.
    """
    if time_to_cyclic:
        dataset['hours_sin'] = np.sin(dataset.hours*(2.*np.pi/24))
        dataset['hours_cos'] = np.cos(dataset.hours*(2.*np.pi/24))
        dataset['days_sin'] = np.sin((dataset.days-1)*(2.*np.pi/7))
        dataset['days_cos'] = np.cos((dataset.days-1)*(2.*np.pi/7))
        dataset['Weeks_sin'] = np.sin((dataset.Weeks-1)*(2.*np.pi/52))
        dataset['Weeks_cos'] = np.cos((dataset.Weeks-1)*(2.*np.pi/52))    
        dataset = dataset.drop(["hours","days","Weeks"],axis = 1)
    """
           
     
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    # RETURN
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

    return col_transform, Xout, Xout2, features_names, LABEL
    