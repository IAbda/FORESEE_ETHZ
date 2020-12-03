# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 22:42:12 2020

@author: imada
"""

#%%
# REF: https://www.tfzx.net/index.php/article/2719468.html
#

#%%
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from treeinterpreter import treeinterpreter as ti
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import plot_roc_curve
from sklearn.inspection import permutation_importance

#%%


"""
-----------------------------------------------------------------
0. IMPORT THE DATA

Import the data with Panda
-----------------------------------------------------------------
"""


# All columns
COLUMNS = ["U", "Ti", "My_Mean", "My_STD", "My_Kurtosis", "My_SKMean", "My_SKKurtosis", "GenRPM_Mean", "GenRPM_STD", "Fault"]

dataset = pd.read_csv("./Data/Faults_Pitch_Yaw.csv", skipinitialspace=True,skiprows=1, sep=',', names=COLUMNS)
print(dataset)

# Input
features_names = dataset.columns.values[0:-1]

# Output: FAULT CLASS LABELS
# (0) - NoFault 
# (1) - YawErrorCorrected
# (2) - YawActuatorStuck
# (3) - PitchSensorFault

LABEL    = "Fault"	
class_names = ['NoFault', 'YawErrorCorrected', 'YawActuatorStuck', 'PitchSensorFault']




#%%
"""
-----------------------------------------------------------------
1. SPLIT THE DATA INTO TRAINING AND TESTING SETS

-----------------------------------------------------------------
"""

X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


#%% Train a base RF, no hyper-parameters optimization here for demo purpose
n_trees = 3;
rfc_base = RandomForestClassifier(n_estimators=n_trees,
                               random_state=42, max_depth=3)
rfc_base.fit(X_train, y_train)
y_pred_base = rfc_base.predict(X_test)

print(rfc_base.predict_proba(X_test))
#Prediction metrics scores:
print(confusion_matrix(y_test,y_pred_base))
print(classification_report(y_test,y_pred_base))
print(accuracy_score(y_test, y_pred_base))
print('-------------------------------------------------\n')


#%% PLOT A SINGLE TREE

sing_tree_id = 2;

# Extract single tree
single_tree_estimator_in_RF = rfc_base.estimators_[sing_tree_id]

from sklearn.tree import export_graphviz

# Export as dot file
export_graphviz(single_tree_estimator_in_RF, out_file='tree.dot', 
                feature_names = features_names,
                class_names = class_names,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
# Copy dot file into: http://www.webgraphviz.com/ Get it rendered there.
# OR
# on Windows: command line: dot tree.dot -Tpng -o tree.png
import os
os.system('dot tree.dot -Tpng -o tree.png')

# on Linux: 
# from subprocess import call
# call(['dot', '-Tpng', '.\tree.dot', '-o', '.\tree.png', '-Gdpi=600'])



#%%
# The decision estimator has an attribute called tree_  which stores the entire
# tree structure and allows access to low level attributes. The binary tree
# tree_ is represented as a number of parallel arrays. The i-th element of each
# array holds information about the node `i`. Node 0 is the tree's root. NOTE:
# Some of the arrays only apply to either leaves or split nodes, resp. In this
# case the values of nodes of the other type are arbitrary!
#
# Among those arrays, we have:
#   - left_child, id of the left child of the node
#   - right_child, id of the right child of the node
#   - feature, feature used for splitting the node
#   - threshold, threshold value at the node
#

# Using those arrays, we can parse the tree structure:

#n_nodes = estimator.tree_.node_count
n_nodes_ = [t.tree_.node_count for t in rfc_base.estimators_]
children_left_ = [t.tree_.children_left for t in rfc_base.estimators_]
children_right_ = [t.tree_.children_right for t in rfc_base.estimators_]
feature_ = [t.tree_.feature for t in rfc_base.estimators_]
threshold_ = [t.tree_.threshold for t in rfc_base.estimators_]


    
#%% ROOT CAUSE TRACER
def explore_tree(estimator, n_nodes, children_left,children_right, feature,threshold,
                suffix='', print_tree= False, sample_id=0, feature_names=None):

    if not feature_names:
        feature_names = feature


    assert len(feature_names) == X.shape[1], "The feature names do not match the number of features."
    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)

    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes"
          % n_nodes)
    if print_tree:
        print("Tree structure: \n")
        for i in range(n_nodes):
            if is_leaves[i]:
                print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
            else:
                print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                      "node %s."
                      % (node_depth[i] * "\t",
                         i,
                         children_left[i],
                         feature[i],
                         threshold[i],
                         children_right[i],
                         ))
            print("\n")
        print()

    # First let's retrieve the decision path of each sample. The decision_path
    # method allows to retrieve the node indicator functions. A non zero element of
    # indicator matrix at the position (i, j) indicates that the sample i goes
    # through the node j.

    node_indicator = estimator.decision_path(X_test)

    # Similarly, we can also have the leaves ids reached by each sample.

    leave_id = estimator.apply(X_test)

    # Now, it's possible to get the tests that were used to predict a sample or
    # a group of samples. First, let's make it for the sample.

    #sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]

    print(X_test[sample_id,:])

    print('Rules used to predict sample %s: ' % sample_id)
    for node_id in node_index:
        # tabulation = " "*node_depth[node_id] #-> makes tabulation of each level of the tree
        tabulation = ""
        if leave_id[sample_id] == node_id:
            print("%s==> Predicted leaf index \n"%(tabulation))
            #continue

        if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print("%sdecision id node %s : (X_test[%s, '%s'] (= %s) %s %s)"
              % (tabulation,
                 node_id,
                 sample_id,
                 feature_names[feature[node_id]],
                 X_test[sample_id, feature[node_id]],
                 threshold_sign,
                 threshold[node_id]))
    print("%sPrediction for sample %d: %s"%(tabulation,
                                          sample_id,
                                          estimator.predict(X_test)[sample_id]))

    # For a group of samples, we have the following common node.
    sample_ids = [sample_id]
    common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) ==
                    len(sample_ids))

    common_node_id = np.arange(n_nodes)[common_nodes]

    print("\nThe following samples %s share the node %s in the tree"
          % (sample_ids, common_node_id))
    print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))

    for sample_id_ in sample_ids:
        print("Prediction for sample %d: %s"%(sample_id_,
                                          estimator.predict(X_test)[sample_id_]))
        
 
        
#%% EXPLORE TREEs IN THE FOREST FOR A GIVEN INPUT RECORD
s_id = 1 # Diagnose this input by exploring the tree branches

for i,e in enumerate(rfc_base.estimators_):

    print("Tree %d\n"%i)
    explore_tree(rfc_base.estimators_[i],n_nodes_[i],children_left_[i],
                 children_right_[i], feature_[i],threshold_[i],
                suffix=i, sample_id=s_id, feature_names=["Feature_%d"%i for i in range(X.shape[1])])
    print('\n'*2)


print(rfc_base.predict_proba(X_test[s_id].reshape(1, -1)))


#%%
#RF TREE INTERPRETER

# Breakdown of feature contributions:
prediction, bias, contributions = ti.predict(rfc_base, X_test[s_id].reshape(1, -1))
print("Prediction", prediction)
print("Bias (trainset prior)", bias)

# Feature contributions
# Calculate the contributors to predicting the class labels, which had the 
# largest impact on updating the prior

#For each class label, calculate the features contribution
print("Feature contributions:")
for c, feature in zip(contributions[0], features_names):
    print(feature, c)

print('\n')
print(bias + np.sum(contributions, axis=1))
    