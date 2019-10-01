'''
You need to carefully initialize X and y as described previously.
Furthermore, inner_cv and outer_cv must be initialized using GroupKFold.
In the parameter_grid, you will need to pass SVM parameters and the
number of PCA components.
'''

import numpy as np
import pandas as pd
from sklearn.svm import SVC   # Support Vector Classifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler   # subtracts mean from each feature and scales to unit var
from sklearn.pipeline import Pipeline
#from BoxClassifier import BoxClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedKFold#, GroupKFold

## Pipeline: Scale, PCA or SVD, then the estimator (i.e. SVM or RandomTree)
#('scaler': StandardScaler())
#('pca', PCA())
steps = [('scaler', StandardScaler()), ('pca', PCA()), ('SVM', SVC())]
pipeline = Pipeline(steps)  # can use make_pipeline instead

# K-fold Nested CV
#     inner_cv = KFold(n_splits=4, shuffle=True, random_state=0)
#     outer_cv = KFold(n_splits=4, shuffle=True, random_state=0)

# Stratified K-fold Nested CV
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# GridSearch params: PCA and SVM optimized with gamma and C params
#    'SVM__C': [0.001, 0.1, 10, 100, 10e5],
#    'SVM__gamma': [0.1, 0.01]
n_components = 150  # image n_components reduction
params = {
    'pca__n_components': [n_components],
    'SVM__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'SVM__gamma': [0.001, 0.01, 0.1, 1, 10, 100]
}
parameter_grid = ParameterGrid(params)

# ================================================
# First Implementation: Single Dimension CV
# ================================================

# Divide the dataset into training and testset with random state and stratified K-fold
# Random state seed gives us reproducable random number (repeats training)
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=30, stratify=Y)

# GridSearch for finding the best params for modeling
grid = GridSearchCV(pipeline, param_grid=params, cv=5) #return_train_score=True)
grid.fit(X_train, y_train)

#scores = nested_cv(X, y, groups, inner_cv, outer_cv, BoxClassifier, parameter_grid)
#print("Cross-validation scores: {}".format(scores))