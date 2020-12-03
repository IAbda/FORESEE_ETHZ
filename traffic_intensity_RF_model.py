# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:49:26 2020

@author: imada
"""

"""
Perform a classification/regression using a Random Forest:
- We predict a target traffic intensity level in a unit location at a specific time interval. 
- We adopt the following types of features: 1) time features, such as hour, day-of-week,
  and week; 2) spatial features, such as location_id; 3) rain (precipitation);   
  4) traffic features such as average hourly traffic speed
  5) Context such as holidays, sporting events, construction, etc.  
"""

#%% IMPORT LIBRARIES

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from treeinterpreter import treeinterpreter as ti
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import plot_roc_curve
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance

#%% INITIALIZE SOME VARIABLES

feature_scaling = False


#%% IMPORT THE DATA

"""
-----------------------------------------------------------------
-1. IMPORT THE DATA

Import the data with Panda
-----------------------------------------------------------------
"""

dataset = pd.read_csv("./Data/OutGenTrafficSyntheticSamples.csv")
print(dataset)

	


#%% ONEHOTENCODING

# Get one hot encoding of columns Context
FEATTMP = pd.get_dummies(dataset.Context, prefix='Context')
# Drop column Context as it is now encoded
dataset = dataset.drop('Context',axis = 1)
# Join the encoded dataset
dataset = dataset.join(FEATTMP)

#%%
"""
-----------------------------------------------------------------
0. FEATURE ENGINEER SOME OF THE INPUT PARAMETERS

-----------------------------------------------------------------
"""
# http://blog.davidkaleko.com/feature-engineering-cyclical-features.html
# We map each cyclical variable onto a circle such that the lowest value for
# that variable appears right next to the largest value.
dataset['hours_sin'] = np.sin(dataset.hours*(2.*np.pi/24))
dataset['hours_cos'] = np.cos(dataset.hours*(2.*np.pi/24))
dataset['days_sin'] = np.sin((dataset.days-1)*(2.*np.pi/7))
dataset['days_cos'] = np.cos((dataset.days-1)*(2.*np.pi/7))
dataset['Weeks_sin'] = np.sin((dataset.Weeks-1)*(2.*np.pi/52))
dataset['Weeks_cos'] = np.cos((dataset.Weeks-1)*(2.*np.pi/52))


#%% ARRANGE COLUMS
ytmp = dataset.traffic_intensity
# Drop column traffic_intensity
dataset = dataset.drop('traffic_intensity',axis = 1)
# Join it at end of dataset
dataset = dataset.join(ytmp)

dataset = dataset.drop(["hours","days","Weeks"],axis = 1)


# Input
features_names = dataset.columns.values[0:-1]

# Output: traffic_intensity estimate
LABEL    = "traffic_intensity"

#%%
"""
-----------------------------------------------------------------
1. SPLIT THE DATA INTO TRAINING AND TESTING SETS

-----------------------------------------------------------------
"""

X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)



#%%
"""
-----------------------------------------------------------------
2. FEATURE SCALING

We know our dataset is not yet a scaled value, for instance the RPM field has 
values in the range of thousands while U has values in range of tens. Therefore, 
it would be beneficial to scale our data (although, this step isn't as important 
for the random forests algorithm). 
-----------------------------------------------------------------
"""


# (OPTIONAL) Feature Scaling
if feature_scaling:
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)




#%%
"""
-----------------------------------------------------------------
3. TRAINING THE RANDOM FOREST CLASSIFIER

-----------------------------------------------------------------
"""
	
# The parameters of the estimator used to apply these methods are optimized by 
# cross-validated search over parameter settings

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 1, stop = 500, num = 10)]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 50, num = 10)]
# max_depth.append(None)
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, True]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_rgr_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_rgr_random.fit(X_train, y_train)


# max_depth = 30
# rf_rgr_random = RandomForestRegressor(n_estimators=100, max_depth=max_depth,random_state=2)
# rf_rgr_random.fit(X_train, y_train)

#%%
bp = rf_rgr_random.best_params_



#%%
"""
-----------------------------------------------------------------
4. EVALUATING THE RANDOM FOREST REGRESSION

-----------------------------------------------------------------
"""

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Traffic Intensity Error: {:0.4f}.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy


best_rfc = rf_rgr_random.best_estimator_
best_rfc_accuracy = evaluate(best_rfc, X_test, y_test)


# Predict on new data
y_pred = rf_rgr_random.predict(X_test)

# Prediction metrics scores:
print("RF train accuracy: %0.3f" % rf_rgr_random.score(X_train, y_train))
print("RF test accuracy: %0.3f" % rf_rgr_random.score(X_test, y_test))

#%%
# Plot the results
plt.figure()
s = 50
a = 0.4
plt.scatter(y_pred, y_test, edgecolor='k',
            c="c", s=s, marker=".", alpha=a,
            label="RF score=%.2f" % rf_rgr_random.score(X_test, y_test))
plt.xlabel("Traffic Intensity: Target")
plt.ylabel("Traffic Intensity: Predictions")
plt.show()

#%% FEATURE IMPORTANCE 
# FOR TRAINING SET

rf_rgr = RandomForestRegressor(n_estimators=500, random_state=42)
rf_rgr.fit(X_train, y_train)
print("Accuracy on train data: {:.2f}".format(rf_rgr.score(X_train, y_train)))

result = permutation_importance(rf_rgr, X_train, y_train, n_repeats=5, n_jobs=-1, random_state=42)
 
perm_sorted_idx = result.importances_mean.argsort()

tree_importance_sorted_idx = np.argsort(rf_rgr.feature_importances_)
tree_indices = np.arange(0, len(rf_rgr.feature_importances_)) + 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
ax1.barh(tree_indices,
         rf_rgr.feature_importances_[tree_importance_sorted_idx], height=0.7)
ax1.set_yticklabels(features_names[tree_importance_sorted_idx])
ax1.set_yticks(tree_indices)
ax1.set_ylim((0, len(rf_rgr.feature_importances_)))
ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False,
            labels=features_names[perm_sorted_idx])
fig.tight_layout()
plt.show()


#%% FEATURE IMPORTANCE 
# FOR TEST SET

rf_rgr.fit(X_test, y_test)
print("Accuracy on test data: {:.2f}".format(rf_rgr.score(X_test, y_test)))

result = permutation_importance(rf_rgr, X_test, y_test, n_repeats=5, n_jobs=-1, random_state=42)
 
perm_sorted_idx = result.importances_mean.argsort()

tree_importance_sorted_idx = np.argsort(rf_rgr.feature_importances_)
tree_indices = np.arange(0, len(rf_rgr.feature_importances_)) + 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
ax1.barh(tree_indices,
         rf_rgr.feature_importances_[tree_importance_sorted_idx], height=0.7)
ax1.set_yticklabels(features_names[tree_importance_sorted_idx])
ax1.set_yticks(tree_indices)
ax1.set_ylim((0, len(rf_rgr.feature_importances_)))
ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False,
            labels=features_names[perm_sorted_idx])
fig.tight_layout()
plt.show()