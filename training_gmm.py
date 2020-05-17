#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:03:55 2019

@author: Pedro, Rodrigo, Alix
"""

import pandas as pd
import numpy as np
import random
# matplotlib inline

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')

#cmap = plt.get_cmap('inferno')

data = pd.read_csv('features/data_selected.csv')
#data = pd.read_csv('../features_csv/data_reagge_hiphop.csv')

labels = np.unique(data['label'])

# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)

genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

conf_mat_total = []
conf_mat_test_total = []

for seed_init in range(100):

    #Initialization of seeds to reproduce results
    np.random.seed(seed_init)
    random.seed(seed_init)

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)

    conf_mat_test = []

    for train_index, test_index in sss.split(X, y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        n_classes = len(np.unique(y_train))

        # Try GMMs using different types of covariances.
        estimator = GaussianMixture(n_components=n_classes,
                      covariance_type='full', max_iter=20, random_state=0)

        #Spherical = variance is the same along all axes, which is not true;
        #Tied = All gaussians share the same covariance matrix, which is not true;
        #Diagonal = Assumes the features are independent from each other, may be interesting
        #Full = Everyone has it's own covariance matrix


        # Since we have class labels for the training data, we can
        # initialize the GMM parameters in a supervised manner.
        estimator.means_init = np.array([X_train[y_train == i].mean(axis=0)
                                        for i in range(n_classes)])

        # Train the other parameters using the EM algorithm.
        estimator.fit(X_train)

        y_train_pred = estimator.predict(X_train)
        accuracy_train = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100

        y_test_pred = estimator.predict(X_test)
        accuracy_test = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100


        # We apply now the trained model to the whole data to see how it behaves
        y_total = estimator.predict(X)


        # To visualize the result, we have a confusion matrix
        #conf_mat_total.append(confusion_matrix(y_total,y))
        conf_mat_test.append(confusion_matrix(y_test_pred,y_test))

    conf_mat_test_total.append(np.sum(conf_mat_test,axis=0))

# Calculate the mean and the variance of the confusion matrices
conf_mat_test_total_mean = np.mean(conf_mat_test_total,axis=0)
conf_mat_test_total_var = np.var(conf_mat_test_total,axis=0)

# We calculate the accuracy using the diagonal of the confusion matrix (the songs that were correctly classified)
accuracy_test_total_prediction = 100*np.trace(conf_mat_test_total_mean)/np.size(data,0)

# We save the confusion matrix as an CSV output
output_mean = pd.DataFrame(conf_mat_test_total_mean)
output_mean = output_mean.rename({idx:labels[idx] for idx in range(np.size(labels))},axis='columns')
output_mean = output_mean.rename({idx:labels[idx] for idx in range(np.size(labels))},axis='index')

output_mean.to_csv (r'conf_matrix_mean_GMM.csv', index = True, header=True)

output_var = pd.DataFrame(conf_mat_test_total_var)
output_var = output_var.rename({idx:labels[idx] for idx in range(np.size(labels))},axis='columns')
output_var = output_var.rename({idx:labels[idx] for idx in range(np.size(labels))},axis='index')

output_var.to_csv (r'conf_matrix_var_GMM.csv', index = True, header=True)

# Print the result to visualize in Spyder environement
print(output_mean)
print("Total accuracy = ", accuracy_test_total_prediction, "%")
