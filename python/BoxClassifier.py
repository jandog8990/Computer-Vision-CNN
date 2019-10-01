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
    def __init__(self, NumOfPCAcomponents, C, gamma):
        """
        Called when initializing the classifier.
        """
        self.C_ = C
        self.gamma_ = gamma
        self.NumOfPCAcomponents_ = NumOfPCAcomponents
        
        # whiten=True
        self.pca = PCA(NumOfPCAcomponents, svd_solver='randomized')
        self.svm = SVC(gamma=gamma, C=C)

    # Please add PCA and SVM in this template.
    # Make sure to use the given parameters.
    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.
        
        ... call .fit() for PCA and .fit for SVC()
        """
        self.pca.fit(X)
        X_train_pca = self.pca.transform(X)
        self.svm.fit(X_train_pca, y)
        print("X_train shape = " + str(X.shape))
        print("X_train_pca shape = " + str(X_train_pca.shape))
        print("\n")

        return self

    def predict(self, X):
        """
        Predicts the classification of a single box of pixels.
        
        ... apply PCA and .predict() for SVC()
        ... Deterimine Class_result for SVC.
        """
        X_test_pca = self.pca.transform(X)
        y_pred = self.svm.predict(X_test_pca)
        
        return y_pred

    def score(self, X, y):
        """
         Returns the classification accuracy assuming
         balanced datasets (half in each category).
         
        ... Applies predition on X, and compares the results
        ... against the actual values in y.
        """
        X_test_pca = self.pca.transform(X)
        accuracy = self.svm.score(X_test_pca, y)
        
        return accuracy