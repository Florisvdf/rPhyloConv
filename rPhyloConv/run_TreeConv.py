# Imports
import pickle
import struct
import random
import os
import time

import numpy as np
from numpy import unique
import pandas as pd
from array import array as pyarray
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc, mean_absolute_error,accuracy_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit, ShuffleSplit, GridSearchCV, ParameterGrid, train_test_split
from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

from Bio import Phylo

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping

from joblib import Parallel, delayed
import multiprocessing
from copy import deepcopy

from models import build_model
from utils.io import group_by_params
from treebuilding import TreeBuilder



# Constants
np.random.seed(256)
random_seed = 42
n_shuffles = 1
test_data_ratio=0.2
n_cv_folds = 5
train_ratio = 1

tree_path = "input_data/HELUIS.Thyroid.phy_tree.tree" 


def auroc(y_true, y_pred):
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)

# Loading data
print('Data loading')
print('')
X_train_df=pd.read_csv(filepath_or_buffer='input_data/X_train.csv',
                       index_col = 0)
X_train_df.fillna(method='ffill',inplace=True)
print('Finished')

print("Number of features: {}".format(len(X_train_df)))


# Loading positive and negative patients ids
neg_ids = pd.read_csv("input_data/patient_ids_neg.csv").values[:, 0].astype(str)
pos_ids = pd.read_csv("input_data/patient_ids_pos.csv").values[:, 0].astype(str)

# Transposing data
X_train_df = X_train_df.transpose()

# Creating the training set based on the patient ids
X_train_df_pos = X_train_df.loc[pos_ids]
X_train_df_neg = X_train_df.loc[neg_ids]
X_train_df = pd.concat([X_train_df_pos, X_train_df_neg], axis = 0)

# In numpy format
#X_full = X_train_df_total.values
y_binary = np.concatenate((np.zeros(143, dtype = int), np.ones(143, dtype = int)))

# Filtering out features
X = filter_features(X_train_df, occurrence_ratio)

# Assigning label and sample names
label_names = {0: 'aTPO Negative', 1: 'aTPO Positive'}

y_verbal = np.asarray([label_names[i] for i in y_binary], dtype=str)
sample_names = np.asarray(list(X_train_df.index), dtype=str)

# Creating dictionaries for storing feature selection data
selected_features = {}
selected_features["Features"] = []
selected_features["Scores"] = []
selected_features["p_values"] = []

list_feat_names = []
list_feat_importances = []

auc_dict = {}
auc_dict["TPR"] = []
auc_dict["FPR"] = []
auc_dict["Thresholds"] = []
auc_dict["AUC"] = []


# Feature selection leakage

print("----------------------------------------------------------------")
print("Performing univariate feature selection on the entire training set")
print("----------------------------------------------------------------\n")
b = SelectPercentile(f_classif,uni_feat_percent)
b.fit(X,y_binary)
X = X.iloc[:, b.get_support(indices=True)]
y=y_binary
selected_features["Features"] = X.columns
feat_names = selected_features["Features"]
selected_features["Scores"] = b.scores_[b.get_support(indices=True)]
selected_features["p_values"] = b.pvalues_[b.get_support(indices=True)]
print("Number of features after feature selection: {}".format(len(X.columns)))



# Creating a parameter grid and instantiating a CV object

param_grid = {
    'input_shape': [None],
    'n_layers': [1],
    'filters': [10],
    'kernel_size': [(3, 9)],
    'pool_size': [(2, 2)],
    'activation': ['elu'],
    'n_classes': [2],
    'learning_rate': [0.001],
    'loss': ['mse'],
    'dropout': [0.3],
    'batch_size': [64],
    'epochs': [2000],
    'callbacks': [[EarlyStopping(patience=40)]],
    'verbose': [2]
}
'''
param_grid = {
    'input_shape': [None],
    'n_layers': [1],
    'filters': [10, 20],
    'kernel_size': [(3, 9)],
    'pool_size': [(1, 2), (2, 2)],
    'activation': ['elu'],
    'n_classes': [2],
    'learning_rate': [0.001],
    'loss': ['mse'],
    'dropout': [0.3],
    'batch_size': [32, 64],
    'epochs': [2000],
    'callbacks': [[EarlyStopping(patience=40)]],
    'verbose': [2]
}
'''
grid_size = 1
for key, value in param_grid.items():
    grid_size *= len(value)
fit_keys = ['batch_size', 'epochs', 'callbacks', 'verbose']
# Stability selection
StratShufSpl=StratifiedShuffleSplit(n_shuffles,
                                        test_size=test_data_ratio, random_state = random_seed)
# CV
skf=StratifiedKFold(n_splits = num_cv_folds)

###############################################################
# Starting stability selection loop
###############################################################
test_stat_df = pd.DataFrame(index=["AUC", "Weighted MSE", "Params", "Features"], columns=[i+1 for i in range(n_shuffles)])
shuffle_counter = 0

print(X.shape)
print(type(X))

n_values = np.max(y) + 1
labels_oh = np.eye(n_values)[y]

# Creating an AUC dict for plotting AUC curves
auc_dict = {}
auc_dict["TPR"] = []
auc_dict["FPR"] = []
auc_dict["Thresholds"] = []
auc_dict["AUC"] = []

for samples,test in StratShufSpl.split(X, y):
    print("----------------------------------------------------------------")
    print("Beginning stability selection iteration {}".format(shuffle_counter))
    print("----------------------------------------------------------------\n")
    shuffle_counter+=1
    X_train, X_test = X.iloc[samples], X.iloc[test]
    y_train, y_test=y[samples], y[test]

    # Creating a dataframe for storing the results
    cv_list = ["CV_{}_GS_{}".format(str(i+1), str(j+1)) for i in range(num_cv_folds) for j in range(len(ParameterGrid(param_grid)))]
    stat_df = pd.DataFrame(index=["AUC", "Weighted MSE", "Params", "Features"], columns=cv_list)

    # Performing the grid search CV
    n_candidates = num_cv_folds * grid_size
    cv_fold = 0
    candidate_counter = 0
    print('Performing GridSearchCV for {} candidates\n\n'.format(n_candidates))
    for train_index, test_index in skf.split(X_train, y_train):
        print("----------------------------------------------------------------")
        print("Beginning cross validation for candidate number {}".format(candidate_counter))
        print("----------------------------------------------------------------\n")
        #################################################################
        # Select and format training and testing sets
        #################################################################
        cv_fold += 1
        gs_it = 0

        train_X, val_X = X_train.iloc[train_index], X_train.iloc[test_index]
        train_index = samples[train_index]
        test_index = samples[test_index]
        train_y, val_y = labels_oh[train_index,:], labels_oh[test_index,:]
        class_frequencies_train = np.sum(train_y, axis = 0)/len(train_y)
        class_frequencies_val = np.sum(val_y, axis = 0)/len(val_y)
        print('Total size of the training set: {}'.format(len(train_X)))
        print('Total size of the validation set: {}'.format(len(val_X)))
        print('Class frequencies in the training set: {}'.format(class_frequencies_train))
        print('Class frequencies in the validation set: {}\n'.format(class_frequencies_val))
     
        
        # Build tree
        print("----------------------------------------------------------------")
        print("Beginning the tree building procedure")
        print("----------------------------------------------------------------\n")

        tree_builder = TreeBuilder(tree_path)
        tree_builder = tree_builder.fit(train_X, train_y)
        train_X = tree_builder.transform(train_X)
        val_X = tree_builder.transform(val_X) 

        for g in ParameterGrid(param_grid):
            candidate_counter += 1
            print('Fitting candidate number {} in shuffle {} with parameters\n'.format(candidate_counter, shuffle_counter))
            print(g)
            gs_it += 1
            params = g.copy()
            fit_params = {key: g.pop(key) for key in fit_keys}

            num_train_samples = train_X.shape[0]
            num_test_samples = val_X.shape[0]
            tree_row = train_X.shape[1]
            tree_col = train_X.shape[2]

            g['input_shape'] = (tree_row, tree_col)
        
            fit_params['x'] = train_X
            fit_params['y'] = train_y
            fit_params['validation_data'] = (val_X, val_y)
            
            # Seting model parameters and fitting
            model = build_model(**g)
            model.fit(**fit_params)

            # Evaluation
            print('Evaluation:\n')
            val_preds = model.predict(val_X)
            auc = roc_auc_score(val_y, val_preds)
            mse = mean_squared_error(val_y, val_preds)
            #mse, auc = model.evaluate(x = val_X, y = val_y, verbose = 0)
            print('MSE: {}'.format(mse))
            print('AUC: {}\n'.format(auc))
            
            # Storing stats in dataframe
            stat_df.loc["Weighted MSE"]["CV_{}_GS_{}".format(cv_fold, gs_it)] = mse
            stat_df.loc["AUC"]["CV_{}_GS_{}".format(cv_fold, gs_it)] = auc
            stat_df.loc["Params"]["CV_{}_GS_{}".format(cv_fold, gs_it)] = params
            stat_df.loc["Features"]["CV_{}_GS_{}".format(cv_fold, gs_it)] = tree_builder.features

            # Resetting model weights and clearing the session
            tf.keras.backend.clear_session()


    print(stat_df)
    # End of GS_CV
    # Find best params according to lowest weighted MSE
    # Refit model on train + val with best params
    # Report all scores on test
    # Save in test_stat_df

    # Saving the results
    try:
        os.mkdir('output_data')
    except OSError as error:
        print('Directory already exists')

    stat_df.to_csv('output_data/validation_results_{}.csv'.format(shuffle_counter))


    # Reftting on the entire training set with the best found parameters
    print('Reftting on the entire training set with the best found parameters\n')
    tf.keras.backend.clear_session()
    grouped_df, params = group_by_params(stat_df, num_combinations = grid_size)
    grouped_df.to_csv('output_data/grouped_validation_results_{}.csv'.format(shuffle_counter))
    best_score_index = np.argmin(list(grouped_df.loc['Weighted MSE']))
    best_params = params[best_score_index]
    print('Best found parameters:\n')
    print(best_params)
    X_train = np.log(X_train + 1)
    X_test = np.log(X_test + 1)
    y_train, y_test = labels_oh[samples,:], labels_oh[test,:]

    
    # Build tree
    tree_builder = TreeBuilder(tree_path)
    tree_builder = tree_builder.fit(X_train, y_train)
    X_train = tree_builder.transform(X_train)
    X_test = tree_builder.transform(X_test) 

    num_train_samples = X_train.shape[0]
    num_test_samples = X_test.shape[0]
    tree_row = X_train.shape[1]
    tree_col = X_train.shape[2]

    fit_params = {key: best_params.pop(key) for key in fit_keys}

    # Splitting the data in train and val for early stopping
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                    stratify=y_train, 
                                                   test_size=0.1)
    fit_params['x'] = X_train
    fit_params['y'] = y_train
    fit_params['validation_data'] = (X_val, y_val)
    best_params['input_shape'] = (tree_row, tree_col)

    print('Samples X_train: {}'.format(len(X_train)))
    print('Samples X_test: {}'.format(len(X_test)))
    print('Labels in y_train: {}\n'.format(np.sum(y_train, axis = 0)))

    #test_untrained_weights = model.get_weights().copy()
    model = build_model(**best_params)
    print('Rebuilt model and thus reinitialized')
    model.fit(**fit_params)
    
    preds = model.predict(X_test)
    np.save('test_preds.npy', preds)
    np.save('test_y.npy', y_test)

    ### ONLY FOR A TEST
    np.save("X_train.npy", X_train)
    model.save("test_model")

    final_params = {**best_params, **fit_params}
    del final_params['x']
    del final_params['y']
    del final_params['validation_data']

    mse = model.evaluate(x = X_test, y = y_test, verbose = 0)
    auc = roc_auc_score(y_test, preds)

    test_stat_df.loc["Weighted MSE"][shuffle_counter] = mse
    test_stat_df.loc["AUC"][shuffle_counter] = auc
    test_stat_df.loc["Features"][shuffle_counter] = tree_builder.features
    test_stat_df.loc["Params"][shuffle_counter] = final_params

    # Storing the FPR, TPR and thresholds for creating an AUC plot
    y_test = y_test[:, 0]
    preds = preds[:, 0]
    fpr, tpr, thresholds = roc_curve(y_test, preds, pos_label=1)
    auc_dict["FPR"].append(fpr)
    auc_dict["TPR"].append(tpr)
    auc_dict["Thresholds"].append(thresholds)
    auc_dict["AUC"].append(auc)


    # Resetting model weights and clearing the session
    tf.keras.backend.clear_session()


# Plotting the AUC curves

for i in range(n_shuffles):
    plt.plot(auc_dict["FPR"][i], auc_dict["TPR"][i],
             label = "Run {} (AUC = {:f})".format(i, auc_dict["AUC"][i]),
             lw=2,
             alpha=0.7)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig("output_data/roc_curve_leakage.pdf")
    
# Saving test results
test_stat_df.to_csv('output_data/test_results.csv')
print('Finished!')