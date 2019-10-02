#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 00:45:21 2019

Class for implementing your classifier for SciKit Learn.
Initial code taken from:
    http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/

@author: alejandrogonzales
"""
#from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
#from sklearn.svm import SVC


class BoxClassifier(BaseEstimator, ClassifierMixin):
    """
    BoxClassifier for distinguishing faces from non-faces
    
    TODO: Implement StandardScaler for standardizing the data before PCA
    """
    
    # Initialize the Pipeline (contains PCA and SVM objects)
    pipeline = None
    pca__n_components = None
    SVM__C = None
    SVM__gamma = None

    # Initialize by passing the parameters of your model:
    def __init__(self, pipeline, pca__n_components, SVM__C, SVM__gamma):
        """
        Called when initializing the classifier.
        """
        self.pipeline = pipeline
        self.pca__n_components = pca__n_components
        self.SVM__C = SVM__C
        self.SVM__gamma = SVM__gamma

    # Please add PCA and SVM in this template.
    # Make sure to use the given parameters.
    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.
        
        ... call .fit() for PCA and .fit for SVC()
        """
        #self.svm.fit(X, y)
        print("PCA n_components = " + str(self.pca__n_components))
        print("SVM C = " + str(self.SVM__C))
        print("SVM gamma = " + str(self.SVM__gamma))
        print("\n")
        self.pipeline.set_params(pca__n_components=self.pca__n_components,
                                 pca__svd_solver='randomized', pca__whiten=True,
                                 SVM__C=self.SVM__C, SVM__gamma=self.SVM__gamma).fit(X,y)

        return self

    def predict(self, X):
        """
        Predicts the classification of a single box of pixels.
        
        ... apply PCA and .predict() for SVC()
        ... Deterimine Class_result for SVC.
        """
        y_pred = self.pipeline.predict(X)
        
        return y_pred

    def score(self, X, y):
        """
         Returns the classification accuracy assuming
         balanced datasets (half in each category).
         
        ... Applies predition on X, and compares the results
        ... against the actual values in y.
        """
        accuracy = self.pipeline.score(X, y)
        
        return accuracy
    
    def transform(self, X):
        """
        Transform the data using PCA class to reduce
        dimensionality of the incoming data set
        """
        xtrans = self.pipeline.transform(X)
        
        return xtrans
