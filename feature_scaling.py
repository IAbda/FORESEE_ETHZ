# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:54:09 2021

@author: imada
"""

#%% IMPORT LIBRARIES

from sklearn import preprocessing


#%%
"""
-----------------------------------------------------------------
FEATURE SCALING:
We know our dataset is not yet a scaled value. Therefore, 
it would be beneficial to scale our data (although, this step isn't as important 
for the random forests algorithm). 
-----------------------------------------------------------------
"""

# (OPTIONAL) Feature Scaling
def feature_scaling(X):
    scaled_X = preprocessing.normalize(X)
    # sc = preprocessing.StandardScaler()
    # scaled_X = sc.fit_transform(X)
    return scaled_X
