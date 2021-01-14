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

#%% IMPORT LIBRARIES

from time import time
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import plot_roc_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline
from scipy.stats import randint
import pickle

from import_dataset import import_dataset
from convert_csv_to_json import make_json
from plot_map import make_plot_map
from load_parse_json import load_parse_json
from predict_from_saved_RF_model import predict_from_saved_RF_model
from save_RF_model_to_disk import save_RF_model_to_disk
from feature_engineer_input import feature_engineer_input
from initialize_vars import initialize_vars
from feature_scaling import feature_scaling




#%% Prediction metrics scores       
"""
-----------------------------------------------------------------

Prediction metrics scores       

-----------------------------------------------------------------
"""

def Average_Traffic_Intensity_Error(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance:')
    print('Average Traffic Intensity Error: {:0.4f}.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy






#%% SPLIT THE DATA INTO TRAINING AND TESTING SETS
"""
-----------------------------------------------------------------

SPLIT THE DATA INTO TRAINING AND TESTING SETS

-----------------------------------------------------------------
"""

def split_data_test_train(dataset, test_size):
    X = dataset.iloc[:, 0:-1].values
    y = dataset.iloc[:, -1].values
    
    n_features=X.shape[1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    
    # CHECK DISTRIBUTIONS OF TRAIN AND TEST SETS
    # Assign colors for each airline and the names
    colors = ['#E69F00', '#56B4E9']
    names = ['y_train', 'y_test']
             
    # Make the histogram using a list of lists
    # Normalize the flights and assign colors and names
    plt.hist([y_train, y_test], bins = 100, density =True,color = colors, label=names)
    
    # Plot formatting
    plt.legend()
    plt.ylabel('Normalized Traffic Intensity')
    plt.title('Side-by-Side Normalized Histogram with y_train & y_test')

    return X, y, X_train, X_test, y_train, y_test




#%% RANDOM FOREST REGRESSOR WITH DEFAULT PARAMETERS
"""
-----------------------------------------------------------------

RANDOM FOREST REGRESSOR WITH DEFAULT PARAMETERS

-----------------------------------------------------------------
"""
# https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py

def RF_Regressor(X_train,X_test,y_train,y_test):
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
    # print("DONE TRAINING: RandomForestRegressor took %.2f seconds to train a model" % (time() - start))
    print("DONE")
    
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

    return rf_rgr




#%% FEATURE IMPORTANCE 
"""
-----------------------------------------------------------------
FEATURE IMPORTANCE OF CLASSIFIER ON TEST DATA

-----------------------------------------------------------------
"""

def feature_importance(model, X_test, y_test, features_names):
    # Using permutation_importance: FOR TEST SET
    # https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py
    
    result = permutation_importance(model, X_test, y_test, n_repeats=5, n_jobs=-1, random_state=42)
     
    perm_sorted_idx = result.importances_mean.argsort()
    
    tree_importance_sorted_idx = np.argsort(model.feature_importances_)
    tree_indices = np.arange(0, len(model.feature_importances_)) + 0.5
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.barh(tree_indices,
             model.feature_importances_[tree_importance_sorted_idx], height=0.7)
    ax1.set_yticklabels(features_names[tree_importance_sorted_idx])
    ax1.set_yticks(tree_indices)
    ax1.set_ylim((0, len(model.feature_importances_)))
    ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False,
                labels=features_names[perm_sorted_idx])
    fig.tight_layout()
    plt.show()



#%% CROSS-VALIDATION
"""
-----------------------------------------------------------------
CROSS-VALIDATION WITH TIME SPLITS ON RandomForestRegressor

-----------------------------------------------------------------
"""

def RF_Regressor_cross_validate(model,X,y,cv):  
    pipeline = Pipeline([
        ('model', model)])
    
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

def RF_Regressor_randomizedsearch(X_train,X_test,y_train,y_test,cv):
    # I will use randomizedsearch to tune my hyperparameters for the Random Forest model.
    
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
    
    return best_rfc_random



    
#%% CROSS-VALIDATION WITH BEST MODEL from RandomizedSearchCV

def RF_Regressor_randomizedsearch_cross_validate(model,X,y,cv):
    pipeline_rf_rs = Pipeline([
        ('model', model)])
    
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
    


#%% MAIN

def main():
    
    # Specify input data file:  
    #   Original client data is specified in a CSV file. 
    #   Clients are familiar with such a file type and format
    csvFilePath = "./data/OutGenTrafficSyntheticSamples.csv"
    # We will convert the csv file to json file
    jsonFilePath = "./data/OutGenTrafficSyntheticSamples.json"
    
    # initialize internal variables
    print('\n')
    print('--- Initialize variables')
    do_feature_scaling, time_to_cyclic, n_splits, n_locations = initialize_vars()

    # Call the make_json function 
    # Convert the csv to json
    print('\n')
    print('--- Convert raw CSV to json')
    make_json(csvFilePath, jsonFilePath)
        
    # Import dataset from json file, and convert to dataframe
    print('\n')
    print('--- Import dataset')
    dataset = load_parse_json(jsonFilePath)
                
    # Feature engineer input features
    print('\n')
    print('--- Feature engineer input features')
    dataset, features_names, LABEL = feature_engineer_input(dataset, time_to_cyclic, do_feature_scaling, n_locations)

    # Split features into test and train sets
    print('\n')
    print('--- Split features into test and train sets')
    test_size = 0.3;
    X, y, X_train, X_test, y_train, y_test = split_data_test_train(dataset, test_size)

    # Train a Random Forest regressor with default parameters
    print('\n')
    print('--- Train Random Forest regressor with default parameters')    
    RF_model = RF_Regressor(X_train,X_test,y_train,y_test)

    # save the model to disk with pickle (other options are possible, but wont bother...)
    print('\n')
    # print('--- Save RF model to disk: Random Forest regressor with default parameters')    
    print('--- Save RF model to disk')    
    saved_model_filename = './saved_models/saved_RF_model.sav'
    save_RF_model_to_disk(RF_model,saved_model_filename)


    # # Feature importance 
    # print('\n')
    # print('--- Feature importance')    
    # feature_importance(RF_model, X_test, y_test, features_names)
    
    # print('Cross-validation of Random Forest regressor')    
    # # Cross-validation approach
    # # cv = KFold(n_splits)
    # cv = StratifiedKFold(n_splits,shuffle=True,random_state=42)
    # # cv = TimeSeriesSplit(n_splits=n_splits) # creating a timeseries split of the datasets

    # # Cross-validation of the Random Forest regressor with default parameters
    # RF_Regressor_cross_validate(RF_model,X,y,cv) 

    # # best RF model with randomizedsearch
    # print('best RF model with randomizedsearch')    
    # best_rfc_random = RF_Regressor_randomizedsearch(X_train,X_test,y_train,y_test,cv)

    # print('Cross-validation of best RF model with randomizedsearch')    
    # RF_Regressor_randomizedsearch_cross_validate(best_rfc_random,X,y,cv)


#%% RUN MAIN
if __name__ == "__main__":

    # Call main function
    main()
    

    

