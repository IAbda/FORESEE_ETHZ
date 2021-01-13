# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:44:39 2021

@author: imada
"""


#%% IMPORT LIBRARIES



#%% INITIALIZE SOME VARIABLES
"""
-----------------------------------------------------------------

INITIALIZE SOME VARIABLES      

-----------------------------------------------------------------
"""

def initialize_vars():
    # global shit is probably a recipe for future disaster, but whatever...
    global do_feature_scaling, time_to_cyclic, n_splits, n_locations
    do_feature_scaling = True
    time_to_cyclic = False
    n_splits = 10; # FOR CROSS-VALIDATION
    n_locations = 10; # NUMBER OF ROAD SECTIONS
    
    return do_feature_scaling, time_to_cyclic, n_splits, n_locations