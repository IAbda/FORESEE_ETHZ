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
  5) Context such as holidays, sporting events, construction, accidents, etc.  
"""
#%%
%reset
%clear

#%% IMPORT LIBRARIES

from time import time
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from treeinterpreter import treeinterpreter as ti
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import plot_roc_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline
from scipy.stats import randint


#%% INITIALIZE SOME VARIABLES

do_feature_scaling = True
time_to_cyclic = False


n_splits = 10; # FOR CROSS-VALIDATION
n_locations = 10; # NUMBER OF ROAD SECTIONS


#%% DEFINE FUNCTIONS
"""
-----------------------------------------------------------------
DEFINE FUNCTIONS

-----------------------------------------------------------------
"""

# Prediction metrics scores:       
def Average_Traffic_Intensity_Error(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance:')
    print('Average Traffic Intensity Error: {:0.4f}.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy


# -----------------------------------------------------------------
# FEATURE SCALING:
# We know our dataset is not yet a scaled value. Therefore, 
# it would be beneficial to scale our data (although, this step isn't as important 
# for the random forests algorithm). 
# -----------------------------------------------------------------

# (OPTIONAL) Feature Scaling
def feature_scaling(X):
    scaled_X = preprocessing.normalize(X)
    # sc = preprocessing.StandardScaler()
    # scaled_X = sc.fit_transform(X)
    return scaled_X







#%% IMPORT THE DATA

"""
-----------------------------------------------------------------
IMPORT THE DATA

Import the data with Panda
-----------------------------------------------------------------
"""

dataset = pd.read_csv("./Data/OutGenTrafficSyntheticSamples.csv")
print(dataset)

	





#%%
"""
-----------------------------------------------------------------
FEATURE ENGINEER SOME OF THE INPUT PARAMETERS

-----------------------------------------------------------------
"""


#%% ONE-HOT-ENCODING

# ONE-HOT-ENCODING of categorical features

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



#%% COMPUTE LAGS
# Feature Engineering for Time Series: Lag Features
# 2-week lag per location (7 days per week and)
dataset['lag_2w'] = dataset['traffic_intensity_plus_60min'].shift(24*14*n_locations)
# 1-week lag per location (7 days per week and)
dataset['lag_1w'] = dataset['traffic_intensity_plus_60min'].shift(24*7*n_locations)
# 2 hours lag per location
dataset['lag_2h'] = dataset['traffic_intensity_plus_60min'].shift(2*n_locations)
# 1 hour lag per location
dataset['lag_1h'] = dataset['traffic_intensity_plus_60min'].shift(1*n_locations)
# remove NaN
dataset.dropna(inplace=True)

dataset['diff_w0'] = dataset['lag_2w'] - dataset['lag_1w']



#%% CONVERT TIME TO CYCLES
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





#%% NORMALIZE & STANDARDIZE

if do_feature_scaling:
    dataset[['precipitation_rate_mm','traffic_speed','lag_1w','lag_2h','lag_1h']]  = \
        feature_scaling(dataset[['precipitation_rate_mm','traffic_speed','lag_1w','lag_2h','lag_1h']].to_numpy()) 


#%% REMOVE ANY COLUMNS

# dataset = dataset.drop(["X_ID","Y_ID"],axis = 1)
dataset = dataset.drop(["loc_ID"],axis = 1)
dataset = dataset.drop(["lag_2w"],axis = 1)



#%% RE-ARRANGE COLUMNS

ytmp = dataset.traffic_intensity_plus_60min
# Drop column traffic_intensity
dataset = dataset.drop('traffic_intensity_plus_60min',axis = 1)
# Join it at end of dataset
dataset = dataset.join(ytmp)

# Input
features_names = dataset.columns.values[0:-1]

# Output: traffic_intensity_plus_60min estimate
LABEL    = "traffic_intensity_plus_60min"




#%%
"""
-----------------------------------------------------------------
SPLIT THE DATA INTO TRAINING AND TESTING SETS

-----------------------------------------------------------------
"""

X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

n_features=X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# CHECK DISTRIBUTIONS OF TRAIN AND TEST SETS
# Assign colors for each airline and the names
colors = ['#E69F00', '#56B4E9']
names = ['y_train', 'y_test']
         
# Make the histogram using a list of lists
# Normalize the flights and assign colors and names
plt.hist([y_train, y_test], bins = 100, density =True,color = colors, label=names)

# Plot formatting
plt.legend()
plt.xlabel(LABEL)
plt.ylabel('Normalized Traffic Intensity')
plt.title('Side-by-Side Normalized Histogram with y_train & y_test')







#%% RandomForestRegressor with mostly default parameters
# https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py

start = time()
# initializing the model which is a Random Forest model and uses default hyperparameters
rf_rgr = RandomForestRegressor(n_estimators=100, 
                               random_state=42, 
                               bootstrap=False,
                               max_depth=20,
                               max_features = "auto",
                               min_samples_leaf=4,
                               min_samples_split=2)

rf_rgr.fit(X_train, y_train)
print("DONE TRAINING: RandomForestRegressor took %.2f seconds to train a model" % (time() - start))

# Predict on new data based on rf_rgr
y_pred_rf_rgr = rf_rgr.predict(X_test)

plt.figure()
s = 50
a = 0.4
plt.scatter(y_test, y_pred_rf_rgr, edgecolor='k',
            c="c", s=s, marker=".", alpha=a,
            label="RF score=%.2f" % rf_rgr.score(X_test, y_test))
plt.plot([0, 6000], [0, 6000], color = 'red', linewidth = 2)
plt.xlabel("Traffic Intensity: Target")
plt.ylabel("Traffic Intensity: Predictions")
plt.grid(b=1,which='both',axis='both')
plt.xticks(np.arange(0, 6001, 500))
plt.xlim(0, 6000)
plt.ylim(0, 6000)
plt.title('RandomForestRegressor')
plt.show()

# Prediction metrics scores:
print("RF train accuracy: %0.3f" % rf_rgr.score(X_train, y_train))
print("RF test accuracy: %0.3f" % rf_rgr.score(X_test, y_test))
rf_rgr_accuracy = Average_Traffic_Intensity_Error(rf_rgr, X_test, y_test)

   


#%% FEATURE IMPORTANCE 
"""
-----------------------------------------------------------------
FEATURE IMPORTANCE OF CLASSIFIER

-----------------------------------------------------------------
"""

# Using permutation_importance: FOR TEST SET
# https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py

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






#%%
# cv = KFold(n_splits)
cv = StratifiedKFold(n_splits,shuffle=True,random_state=42)
# cv = TimeSeriesSplit(n_splits=n_splits) # creating a timeseries split of the datasets




#%% CROSS-VALIDATION
"""
-----------------------------------------------------------------
CROSS-VALIDATION WITH TIME SPLITS ON RandomForestRegressor

-----------------------------------------------------------------
"""

pipeline = Pipeline([
    ('model', rf_rgr)])

pipeline_rf_rgr = pipeline.fit(X, y)

# doing cross validation on the chunks of data and calculating scores
scores_rf = cross_validate(pipeline_rf_rgr, X, y, cv=cv,
                          scoring=['neg_mean_squared_error', 'neg_mean_absolute_error',
                                  'neg_mean_squared_log_error'],
                          return_train_score=True, n_jobs=-1)

# root mean squared error
print('Random Forests: Average RMSE train data:', 
      sum([np.sqrt(-1 * x) for x in scores_rf['train_neg_mean_squared_error']])/len(scores_rf['train_neg_mean_squared_error']))
print('Random Forests: Average RMSE test data:', 
      sum([np.sqrt(-1 * x) for x in scores_rf['test_neg_mean_squared_error']])/len(scores_rf['test_neg_mean_squared_error']))

# mean absolute error
print('Random Forests: Average MAE train data:', 
      sum([(-1 * x) for x in scores_rf['train_neg_mean_absolute_error']])/len(scores_rf['train_neg_mean_absolute_error']))
print('Random Forests: Average MAE test data:', 
      sum([(-1 * x) for x in scores_rf['test_neg_mean_absolute_error']])/len(scores_rf['test_neg_mean_absolute_error']))

# root mean squared log error
print('Random Forests: Average RMSLE train data:', 
      sum([(-1 * x) for x in scores_rf['train_neg_mean_squared_log_error']])/len(scores_rf['train_neg_mean_squared_log_error']))
print('Random Forests: Average RMSLE test data:', 
      sum([(-1 * x) for x in scores_rf['test_neg_mean_squared_log_error']])/len(scores_rf['test_neg_mean_squared_log_error']))







#%% GRID AND RANDOM SEARCH OPTIMIZATION OF HYPERPARAMETERS
"""
-----------------------------------------------------------------
OPTIMIZATION OF THE RANDOM FOREST CLASSIFIER WITH RANDOM GRID SEARCH

-----------------------------------------------------------------
"""

# I will use randomizedsearch to tune my hyperparameters for the Random Forest model.

#%% GRID AND RANDOM SEARCH OPTIMIZATION	
# The parameters of the estimator used to apply these methods are optimized by 
# cross-validated search over parameter settings

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 1, stop = 499, num = 24)]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 100, num = 10)]
# max_depth.append(None)
# Number of features to consider at every split
#max_features = ['auto','log2', 'sqrt']
max_features = ['auto']
# Minimum number of samples required to split a node
min_samples_split = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# Method of selecting samples for training each tree
# bootstrap = [True, False]
bootstrap = [False]

# Create the random grid
srch_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
print(srch_grid)


# RandomizedSearchCV

# Use the random grid to search for best hyperparameters
# First create the base model to tune
start = time()
rf = RandomForestRegressor()

# Random search of parameters, using cross validation, 
# search across different combinations, and use all available cores
rf_rgr_random = RandomizedSearchCV(estimator = rf, 
                                    param_distributions = srch_grid, 
                                    n_iter = 5, 
                                    cv = cv, 
                                    verbose=10, 
                                    random_state=42, 
                                    n_jobs = -1)
# Fit the random search model
rf_rgr_random.fit(X_train, y_train)
print("DONE RandomizedSearchCV TRAINING: RandomizedSearchCV took %.2f seconds to train a model" % (time() - start))

best_rfc_random = rf_rgr_random.best_estimator_
print(best_rfc_random)

# Predict on new data based on y_pred_rgr_random
y_pred_rgr_random = rf_rgr_random.predict(X_test)

plt.figure()
s = 50
a = 0.4
plt.scatter(y_test,y_pred_rgr_random, edgecolor='k',
            c="c", s=s, marker=".", alpha=a,
            label="RF score=%.2f" % rf_rgr_random.score(X_test, y_test))
plt.plot([0, 6000], [0, 6000], color = 'red', linewidth = 2)
plt.xlabel("Traffic Intensity: Target")
plt.ylabel("Traffic Intensity: Predictions")
plt.grid(b=1,which='both',axis='both')
plt.xticks(np.arange(0, 6001, 500))
plt.xlim(0, 6000)
plt.ylim(0, 6000)
plt.title('RandomizedSearchCV')
plt.show()


# Prediction metrics scores:
print("RF train accuracy following RandomizedSearchCV: %0.3f" % rf_rgr_random.score(X_train, y_train))
print("RF test accuracy following RandomizedSearchCV: %0.3f" % rf_rgr_random.score(X_test, y_test))
best_rfc_random_accuracy = Average_Traffic_Intensity_Error(best_rfc_random, X_test, y_test)



#%% CROSS-VALIDATION WITH BEST MODEL from RandomizedSearchCV

pipeline_rf_rs = Pipeline([
    ('model', best_rfc_random)])

# doing cross validation on the chunks of data and calculating scores
scores_rf = cross_validate(pipeline_rf_rs, X, y, cv=cv,
                          scoring=['neg_mean_squared_error', 'neg_mean_absolute_error',
                                  'neg_mean_squared_log_error'],
                          return_train_score=True, n_jobs=-1)

# root mean squared error
print('Random Forests: Average RMSE train data:', 
      sum([np.sqrt(-1 * x) for x in scores_rf['train_neg_mean_squared_error']])/len(scores_rf['train_neg_mean_squared_error']))
print('Random Forests: Average RMSE test data:', 
      sum([np.sqrt(-1 * x) for x in scores_rf['test_neg_mean_squared_error']])/len(scores_rf['test_neg_mean_squared_error']))

# absolute mean error
print('Random Forests: Average MAE train data:', 
      sum([(-1 * x) for x in scores_rf['train_neg_mean_absolute_error']])/len(scores_rf['train_neg_mean_absolute_error']))
print('Random Forests: Average MAE test data:', 
      sum([(-1 * x) for x in scores_rf['test_neg_mean_absolute_error']])/len(scores_rf['test_neg_mean_absolute_error']))

# root mean squared log error
print('Random Forests: Average RMSLE train data:', 
      sum([(-1 * x) for x in scores_rf['train_neg_mean_squared_log_error']])/len(scores_rf['train_neg_mean_squared_log_error']))
print('Random Forests: Average RMSLE test data:', 
      sum([(-1 * x) for x in scores_rf['test_neg_mean_squared_log_error']])/len(scores_rf['test_neg_mean_squared_log_error']))






#%%

# from numpy import asarray
# from pandas import read_csv
# from pandas import DataFrame
# from pandas import concat
# from sklearn.metrics import mean_absolute_error
# from sklearn.ensemble import RandomForestRegressor
# from matplotlib import pyplot


# # transform a time series dataset into a supervised learning dataset
# def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
# 	n_vars = 1 if type(data) is list else data.shape[1]
# 	df = DataFrame(data)
# 	cols = list()
# 	# input sequence (t-n, ... t-1)
# 	for i in range(n_in, 0, -1):
# 		cols.append(df.shift(i))
# 	# forecast sequence (t, t+1, ... t+n)
# 	for i in range(0, n_out):
# 		cols.append(df.shift(-i))
# 	# put it all together
# 	agg = concat(cols, axis=1)
# 	# drop rows with NaN values
# 	if dropnan:
# 		agg.dropna(inplace=True)
# 	return agg.values

# # split a univariate dataset into train/test sets
# def train_test_split(data, n_test):
# 	return data[:-n_test, :], data[-n_test:, :]

# # fit an random forest model and make a one step prediction
# def random_forest_forecast(train, testX):
# 	# transform list into array
# 	train = asarray(train)
# 	# split into input and output columns
# 	trainX, trainy = train[:, :-1], train[:, -1]
# 	# fit model
# 	model = RandomForestRegressor(n_estimators=1000)
# 	model.fit(trainX, trainy)
# 	# make a one-step prediction
# 	yhat = model.predict([testX])
# 	return yhat[0]

# # walk-forward validation for univariate data
# def walk_forward_validation(data, n_test):
# 	predictions = list()
# 	# split dataset
# 	train, test = train_test_split(data, n_test)
# 	# seed history with training dataset
# 	history = [x for x in train]
# 	# step over each time-step in the test set
# 	for i in range(len(test)):
# 		# split test row into input and output columns
# 		testX, testy = test[i, :-1], test[i, -1]
# 		# fit model on history and make a prediction
# 		yhat = random_forest_forecast(history, testX)
# 		# store forecast in list of predictions
# 		predictions.append(yhat)
# 		# add actual observation to history for the next loop
# 		history.append(test[i])
# 		# summarize progress
# 		print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
# 	# estimate prediction error
# 	error = mean_absolute_error(test[:, -1], predictions)
# 	return error, test[:, -1], predictions

# # load the dataset
# series = read_csv('./Data/daily-total-female-births.csv', header=0, index_col=0)
# values = series.values
# # transform the time series data into supervised learning
# data = series_to_supervised(values, n_in=6)
# # evaluate
# mae, y, yhat = walk_forward_validation(data, 4)
# print('MAE: %.3f' % mae)
# # plot expected vs predicted
# pyplot.plot(y, label='Expected')
# pyplot.plot(yhat, label='Predicted')
# pyplot.legend()
# pyplot.show()

