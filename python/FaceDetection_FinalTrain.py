#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 00:26:20 2019

Final Classifier training and validation on 80/20 split

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
Final Classifier code to find the optimized model 
for classifying faces and non-faces
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

# K-fold Nested CV
#   inner_cv = KFold(n_splits=5, shuffle=True, random_state=0)
#   outer_cv = KFold(n_splits=5, shuffle=True, random_state=0)

# Stratified K-fold Nested CV with repeatable tests (i.e. SEED=0)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Group K-Fold CV
#inner_cv = GroupKFold(n_splits=5)
#outer_cv = GroupKFold(n_splits=5)

# Final PCA and SVM parameters for optimized model
# PCA and SVM optimized with gamma and C params
n_components = 150  # image n_components reduction
params = {
    'pca__n_components': [n_components],
    'SVM__C': [1e3, 5e3, 1e4, 5e4, 1e5],
    'SVM__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
}
parameter_grid = ParameterGrid(params)

#params = {
#    'pca__n_components': n_components,
#    'SVM__C': 1e3,
#    'SVM__gamma': 0.0001
#}
#parameter_grid = ParameterGrid(params)

# Split the data into 80/20 training and test
# Divide the dataset into training and testset with random state and stratified K-fold
# Random state seed gives us reproducable random number (repeats training)
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=30, stratify=y)

# Pipeline: Scale, PCA or SVD, then the estimator (i.e. SVM or RandomTree)
steps = [('scaler', StandardScaler()), ('pca', PCA()), ('SVM', SVC())]
pipeline = Pipeline(steps)  # can use make_pipeline instead

#pca = PCA(n_components=params['pca__n_components'])
#pipeline.set_params(pca__n_components=params['pca__n_components']).fit(X_train, y_train) #pca__n_components=params['pca__n_components'])

# ====================================
# Final Classifier on 80/20 Split
# ====================================

# Inner cross-validation
nestedCV = NestedCV()
final_score, params, clf = nestedCV.inner_cv(X_train, y_train, X_test, y_test,
                                        inner_cv, outer_cv, BoxClassifier, pipeline, parameter_grid)

# Calculate prediction and test/train accuracy to confirm model params
y_pred = clf.predict(X_test)
acc_test = clf.score(X_test, y_test)
acc_train = clf.score(X_train, y_train)


# TEST: Initialize single BoxClassifier instance with pipeline and parameter grid
#boxClassifier = BoxClassifier(pipeline, **params)
#boxClassifier.fit(X_train, y_train)