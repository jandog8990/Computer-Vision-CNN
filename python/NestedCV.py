# Class for implementing Nested Cross-Validation

import numpy as np
from BoxClassifier import BoxClassifier

class NestedCV:
    """
    NestedCV class for computing outer and inner 
    cross validation for finding best params
    """
    def __init__(self):
        """
        initialize class
        """
        
    # TODO make sure and implement pipeline in this method
    def nested_cv(self, X, y, groups, inner_cv, outer_cv, Classifier, parameter_grid):
        """
        Uses nested cross-validation to optimize and exhaustively evaluate
        the performance of a given classifier. The original code was taken from
        Chapter 5 of Introduction to Machine Learning with Python. However, it
        has been modified.
    
        Input parameters:
           X, y: describe one set of boxes grouped by image number.
    
        Output:
           The function returns the scores from the outer loop.
        """
        outer_scores = []
        # for each split of the data in the outer cross-validation
        # (split method returns indices of training and test parts)
        #
        outerCount = 0
        for training_samples, test_samples in outer_cv.split(X, y, groups):
            # initialize the outter training and testing sets
            X_outer_train = X[training_samples]
            y_outer_train = y[training_samples]
            X_outer_test = X[test_samples]
            y_outer_test = y[test_samples]
            
            # find best parameter using inner cross-validation
            best_params = {}
            best_score = -np.inf
            # iterate over parameters
            for parameters in parameter_grid:
                # accumulate score over inner splits
                cv_scores = []
                
                # iterate over inner cross-validation
                for inner_train, inner_test in inner_cv.split(
                       X[training_samples], y[training_samples],
                       groups[training_samples]):
                    
                    # initialize the inner training and testing data sets
                    X_inner_train = X[inner_train]
                    y_inner_train = y[inner_train]
                    X_inner_test = X[inner_test]
                    y_inner_test = y[inner_test]
                    
                    # Implement BoxClassifier given pipeline, 
                    # parameters and training data
                    clf = Classifier(**parameters)
                        
                    # PCA fit to the inner train data
                    clf.pca_fit(X_inner_train)
                    X_inner_train_pca = clf.transform(X_inner_train)
                   
                    # Classifier fit the inner training data
                    clf.fit(X_inner_train_pca, y_inner_train)
                   
                    # Transform the test data w PCA
                    X_inner_test_pca = clf.transform(X_inner_test)
                    score = clf.score(X_inner_test_pca, y_inner_test)
                    cv_scores.append(score)
                        
                # compute mean score over inner folds
                # for a single combination of parameters.
                mean_score = np.mean(cv_scores)
                if mean_score > best_score:
                    # if better than so far, remember parameters
                    best_score = mean_score
                    best_params = parameters
                    
                # Show the current scores and best params
#                print("Current Training Params:")
#                print(parameters)
#                print("mean_score = " + str(mean_score))
#                print("best_score = " + str(best_score))
#                print("best_params:")
#                print(best_params)
#                print("\n")
            
            print("Final Classifier " + str(outerCount) + ": Best Params:")
            print(best_params)
            
            # Build classifier on best parameters using outer training set
            # This is done over all parameters evaluated through a single
            # outer fold and all inner folds.
            clf = Classifier(**best_params)
            
            # PCA fit to the outer train data
            clf.pca_fit(X_outer_train)
            X_outer_train_pca = clf.transform(X_outer_train)
           
            # Classifier fit the outer training data
            clf.fit(X_outer_train_pca, y_outer_train)
           
            # Transform the test data w PCA
            X_outer_test_pca = clf.transform(X_outer_test)
            score = clf.score(X_outer_test_pca, y_outer_test)
            outer_scores.append(score)
                
            outerCount = outerCount + 1
            print("outer score = " + str(score))
            print("\n")
            
        print("Final Results:")
        print("outer scores = " + str(outer_scores))
        print("\n")
                
        return np.array(outer_scores)
    
    # inner cross validation to get the final parameters and classifier
    def inner_cv(self, X, y, X_test, y_test, inner_cv, outer_cv, Classifier, pipeline, parameter_grid):
        # find best parameter using inner cross-validation
        best_params = {}
        best_score = -np.inf
        # iterate over parameters
        for parameters in parameter_grid:
            # accumulate score over inner splits
            cv_scores = []
            
            # iterate over inner cross-validation
            for inner_train, inner_test in inner_cv.split(X,y):
                
                # initialize the inner training and testing data sets
                X_inner_train = X[inner_train]
                y_inner_train = y[inner_train]
                X_inner_test = X[inner_test]
                y_inner_test = y[inner_test]
                
                # build BoxClassifier given parameters and training data
                clf = Classifier(pipeline, **parameters)
                    
                # PCA fit to the inner train data
                clf.fit(X_inner_train, y_inner_train)
               
                # Score the classifier against test data
                score = clf.score(X_inner_test, y_inner_test)
                cv_scores.append(score)
                    
            # compute mean score over inner folds
            # for a single combination of parameters.
            mean_score = np.mean(cv_scores)
            if mean_score > best_score:
                # if better than so far, remember parameters
                best_score = mean_score
                best_params = parameters
                
            # Show the current scores and best params
            print("Current Training Params:")
            print(parameters)
            print("mean_score = " + str(mean_score))
            print("best_score = " + str(best_score))
            print("best_params:")
            print(best_params)
            print("\n")
        
        print("Final Classifier Best Params:")
        print(best_params)
        
        # Build classifier on best parameters using outer training set
        # This is done over all parameters evaluated through a single
        # outer fold and all inner folds.
        clf = Classifier(pipeline, **best_params)
        
        # PCA fit to the outer train data
        clf.fit(X, y)
       
        # Score the test data
        final_score = clf.score(X_test, y_test)
            
        print("final score = " + str(final_score))
        print("\n")
        
        return final_score, best_params, clf