#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 00:26:20 2019

Load and create data from the GroundTruth anchor pixels
NOTE: This is worked in tandem with the Jupyter code

@author: alejandrogonzales
"""

import numpy as np
import random

# ===========================================================
# ===========================================================

## Import the FaceBoxes and BackBoxes from the LoadCreateAnchors.py
## script which creates face and bg pixel images
#dataDir = '../python/'
#FaceBoxes = np.load(dataDir+'FaceBoxes.npy', allow_pickle=True)
#BackBoxes = np.load(dataDir+'BackBoxes.npy', allow_pickle=True)
#FrameGroup = [i for i in range(len(FaceBoxes))]
#
## Make balanced datasets for faces and BG
#x = [random.randint(0,len(FaceBoxes)) for i in range(len(FaceBoxes)-len(BackBoxes))]
#FaceBoxes = np.delete(FaceBoxes, x, axis=0)
#print(x)
#print(FaceBoxes.shape)
#print(BackBoxes.shape)
#
## Create the datasets and labels for the Face and Back boxes
#X = np.concatenate((FaceBoxes, BackBoxes))
#flabels = [1 for i in range(len(FaceBoxes))]
#blabels = [0 for i in range(len(BackBoxes))]
#y = np.concatenate((flabels, blabels))
#
#'''
#You need to carefully initialize X and y as described previously.
#Furthermore, inner_cv and outer_cv must be initialized using GroupKFold.
#In the parameter_grid, you will need to pass SVM parameters and the
#number of PCA components.
#'''
#
#from sklearn.svm import SVC   # Support Vector Classifier
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler   # subtracts mean from each feature and scales to unit var
#from sklearn.pipeline import Pipeline
##from BoxClassifier import BoxClassifier
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
#
#from sklearn.model_selection import ParameterGrid
#from sklearn.model_selection import StratifiedKFold#, GroupKFold
#
### Pipeline: Scale, PCA or SVD, then the estimator (i.e. SVM or RandomTree)
##('scaler': StandardScaler())
##('pca', PCA())
#steps = [('scaler', StandardScaler()), ('pca', PCA()), ('SVM', SVC())]
#pipeline = Pipeline(steps)  # can use make_pipeline instead
#
## K-fold Nested CV
##     inner_cv = KFold(n_splits=4, shuffle=True, random_state=0)
##     outer_cv = KFold(n_splits=4, shuffle=True, random_state=0)
#
## Stratified K-fold Nested CV
#inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
#outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
#
## GridSearch params: PCA and SVM optimized with gamma and C params
##    'SVM__C': [0.001, 0.1, 10, 100, 10e5],
##    'SVM__gamma': [0.1, 0.01]
#n_components = 150  # image n_components reduction
#params = {
#    'pca__n_components': [n_components],
#    'SVM__C': [0.001, 0.01, 0.1, 1, 10, 100],
#    'SVM__gamma': [0.001, 0.01, 0.1, 1, 10, 100]
#}
##parameter_grid = ParameterGrid(params)
#
## ================================================
## First Implementation: Single Dimension CV
## ================================================
#
## Divide the dataset into training and testset with random state and stratified K-fold
## Random state seed gives us reproducable random number (repeats training)
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.3, random_state=30, stratify=y)
#
## GridSearch for finding the best params for modeling
##return_train_score=True)
#grid = GridSearchCV(pipeline, param_grid=params, cv=5) 
#grid.fit(X_train, y_train)
#
##scores = nested_cv(X, y, groups, inner_cv, outer_cv, BoxClassifier, parameter_grid)
##print("Cross-validation scores: {}".format(scores))

print("Test Score = %3.2f" % (grid.score(X_test, y_test)))
print("Train Score = %3.2f" % (grid.score(X_train, y_train)))
print("Best params = " + str(grid.best_params_))

# Predict output values for input X_test using model from training
y_predict = grid.predict(X_test)
plt.scatter(y_test, y_predict)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")

