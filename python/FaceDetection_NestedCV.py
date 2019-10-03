#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 00:26:20 2019

Load and create data from the GroundTruth anchor pixels

@author: alejandrogonzales
"""

import numpy as np
import random

# Import the FaceBoxes and BackBoxes from the LoadCreateAnchors.py
# script which creates face and bg pixel images
dataDir = '../python/'
FaceBoxes = np.load(dataDir+'FaceBoxes.npy', allow_pickle=True)
BackBoxes = np.load(dataDir+'BackBoxes.npy', allow_pickle=True)
FrameGroup = [i for i in range(len(FaceBoxes))]

# Make balanced datasets for faces and BG
x = [random.randint(0,len(FaceBoxes)) for i in range(len(FaceBoxes)-len(BackBoxes))]
FaceBoxes = np.delete(FaceBoxes, x, axis=0)

# Create the datasets and labels for the Face and Back boxes
X = np.concatenate((FaceBoxes, BackBoxes))
flabels = [1 for i in range(len(FaceBoxes))]
blabels = [0 for i in range(len(BackBoxes))]
y = np.concatenate((flabels, blabels))
group1  = [i for i in range(len(flabels))]
groups = np.concatenate((group1, group1))


'''
You need to carefully initialize X and y as described previously.
Furthermore, inner_cv and outer_cv must be initialized using GroupKFold.
In the parameter_grid, you will need to pass SVM parameters and the
number of PCA components.
'''

from sklearn.svm import SVC   # Support Vector Classifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler   # subtracts mean from each feature and scales to unit var
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt

# Import custom classes for nested cv and BoxClassifier
from NestedCV import NestedCV
from BoxClassifier import BoxClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedKFold, GroupKFold

## Pipeline: Scale, PCA or SVD, then the estimator (i.e. SVM or RandomTree)
steps = [('scaler', StandardScaler()), ('pca', PCA()), ('SVM', SVC())]
pipeline = Pipeline(steps)  # can use make_pipeline instead

# K-fold Nested CV
#   inner_cv = KFold(n_splits=5, shuffle=True, random_state=0)
#   outer_cv = KFold(n_splits=5, shuffle=True, random_state=0)

# Stratified K-fold Nested CV with repeatable tests (i.e. SEED=0)
#   inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
#   outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Group K-Fold CV
inner_cv = GroupKFold(n_splits=5)
outer_cv = GroupKFold(n_splits=5)

# PCA and SVM optimized with gamma and C params
n_components = 150  # image n_components reduction
params = {
    'n_components': [n_components],
    'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
}
parameter_grid = ParameterGrid(params)

# =========================================================
# Final Implementation: Customer Nested Cross-Validation
# =========================================================
# Apply nested cross-validation (wtf does FrameGroup have to do w anything??)
#scores = NestedCV.nested_cv(X, y, FrameGroup, inner_cv, outer_cv, SVC, parameter_grid)
nestedCV = NestedCV()
scores = nestedCV.nested_cv(X, y, groups, inner_cv, outer_cv, BoxClassifier, parameter_grid)
print("\nCross-validation scores: {}".format(scores))

# Predict output values for input X_test using model from training
#y_predict = grid.predict(X_test)
#plt.scatter(y_test, y_predict)
#plt.xlabel("True Values")
#plt.ylabel("Predicted Values")