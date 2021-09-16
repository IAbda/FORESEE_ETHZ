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
    global n_splits, kstepsahead, n_trees, max_depth
    kstepsahead=2 # k hours ahead for prediction, this should be a user input
    n_splits = 10; # FOR CROSS-VALIDATION
    n_trees = 300 #100
    max_depth=20  # 20

    return n_splits, kstepsahead, n_trees, max_depth