#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 00:45:21 2019

Class for implementing your classifier for SciKit Learn.
Initial code taken from:
    http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/

@author: alejandrogonzales
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.svm import SVC


class BoxClassifier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""
    
    # Initialize the PCA and SVM objects
    pca = None
    svm = None

    # Initialize by passing the parameters of your model:
    def __init__(self, n_components, C, gamma):
        """
        Called when initializing the classifier.
        """
        self.C_ = C
        self.gamma_ = gamma
        self.n_components_ = n_components
        
        # whiten=True
        self.pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
        self.svm = SVC(gamma=gamma, C=C)
#        SVC(kernel='rbf', class_weight='balanced')

    # Please add PCA and SVM in this template.
    # Make sure to use the given parameters.
    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.
        
        ... call .fit() for PCA and .fit for SVC()
        """
        self.svm.fit(X, y)

        return self
    
    def pca_fit(self, X):
        """
        PCA dimension reduction using SVD on
        input data X
        """
        self.pca.fit(X)
        
        return self

    def predict(self, X):
        """
        Predicts the classification of a single box of pixels.
        
        ... apply PCA and .predict() for SVC()
        ... Deterimine Class_result for SVC.
        """
        y_pred = self.svm.predict(X)
        
        return y_pred

    def score(self, X, y):
        """
         Returns the classification accuracy assuming
         balanced datasets (half in each category).
         
        ... Applies predition on X, and compares the results
        ... against the actual values in y.
        """
        accuracy = self.svm.score(X, y)
        
        return accuracy
    
    def transform(self, X):
        """
        Transform the data using PCA class to reduce
        dimensionality of the incoming data set
        """
        X_pca = self.pca.transform(X)
        
        return X_pca
