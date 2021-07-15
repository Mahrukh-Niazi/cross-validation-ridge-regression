#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 00:04:49 2021

@author: Mahrukh Niazi
"""
from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Function Definitions:

def shuffle_data(data: tuple) -> tuple:
    """This function takes data as an argument and returns its randomly 
    permutated version along the samples.
    

    Inputs:
        - data: variable corresponding to the target vector (response) and 
                feature design matrix of the dataset, structured as a tuple.
    
    Outputs:
        - reshuffled dataset, maintaining the target and feature pair as a
          tuple of size two: the element at index 0 corresponds to the array
          of target responses and the element at index 1 corresponds to the
          array of feature matrix.

    """
    shuffled = {}
    p = np.random.permutation(len(data['X']))
    shuffled['X'], shuffled['t'] = data['X'][p], data['t'][p]

    return shuffled

def split_data(data: tuple, num_folds: int, fold: int) -> tuple:
    """This function takes data, number of partitions as num_folds, and the
    selected partition fold as its arguments and returns the selected 
    partition fold as data_fold, and the remaining data as data_rest. It 
    splits the dataset (training data) into num_fold - 1 training sets and 1
    validation set for a num_fold-cross validation algorithm. 
    
    Inputs:
        - data: variable corresponding to the target vector (response) and 
                feature design matrix of the dataset, structured as a tuple.
        
        - num_folds: integer corresponding to the k number of folds used for
                     cross validation.
        
        - fold: integer corresponding to the selected fold for the validation
                set.
    Output:
        - a tuple of size two: the element at index 0 corresponds to an
          array consisting of the selected validation set data_fold, and the
          element at index 1 corresponds to the array consisting of the 
          remaining folds for the training set data_rest for a num_folds 
          K-cross validation.
        
    """
    # Calculates the size of each fold
    sz_fold = int(len(data['t'])/num_folds)

    # Calculates the limits of validation fold
    idx_fold = list(range((fold-1)*sz_fold, fold*sz_fold-1))

    # Calculates the difference of the two sets
    idx_rest = list(set(range(0,len(data['t'])))-set(idx_fold))

    # Assign the correct portions of each folder set
    data_fold = {'X':data['X'][idx_fold,:],'t':data['t'][idx_fold]}
    data_rest = {'X':data['X'][idx_rest,:],'t':data['t'][idx_rest]}

    return(data_fold, data_rest)

def train_model(data, lambd: float) -> np.ndarray:
    """This function takes data and lambd as arguments, and returns the 
    coefficients of ridge regression with penalty level lambda.
    
    Inputs:
        - data: a tuple of size two, with element at index 0 corresponding to
                an array of target responses and element at index 1 
                corresponding to an array of feature matrix OR an array 
                corresponding to a specific fold from a pre-split k-fold cross 
                validation dataset. Data is assumed to be centered so no
                intercept is included. 
                
        - lambd: an integer for a specific lambda penalty value.
        
    Output:
        - an array consisting of the coefficient estimates derived from a 
          ridge regression.

    """
    xtx = np.matmul(np.transpose(data['X']), data['X'])
    inverse = np.linalg.inv(xtx + lambd * np.identity(len(data['X'][0])))

    return np.matmul(np.matmul(inverse, np.transpose(data['X'])), data['t'])

def predict(data, model: np.ndarray) -> np.ndarray:
    """This function takes data and model as its arguments, and returns the
    linear regression predicitons based on data and model.
    
    Inputs:
        - data: a tuple of size two, with element at index 0 corresponding to
                an array of target responses and element at index 1 
                corresponding to an array of feature matrix OR an array 
                corresponding to a specific fold from a pre-split k-fold cross 
                validation dataset.
        
        - model: an array consisting of the coefficient estimates derived from 
                 a ridge regression.
    
    Outputs:
        - an array consisting of the predictions derived from a linear ridge
          regression.
    """
    # predictions = np.matmul(data['X'],model)

    return data['X'].dot(model)
        
def loss(data, model: np.ndarray) -> float:
    """This function takes data and model as its arguments and returns the 
    average squared error loss based on model. With the following variables
    defined as: y = target response vector, X = feature design matrix,
    \beta = coefficients estimates vector, and n = sample size, the error loss
    equation is given by MSE = (1/n) ((y - X\beta)^2).
    
    Inputs:
        - data: a tuple of size two, with element at index 0 corresponding to
                an array of target responses and element at index 1 
                corresponding to an array of feature matrix OR an array 
                corresponding to a specific fold from a pre-split k-fold cross 
                validation dataset.
        
        - model: an array consisting of the coefficient estimates derived from 
                 a ridge regression.
    
    Outputs:
        - an array consisting of the MSE error derived from a linear ridge
          regression on the validation set.

    """
    # error = pow(np.linalg.norm(data['t']-np.matmul(data['X'],model)),2)/len(data['t'])

    return np.sum((data['t'] - predict(data, model)) ** 2) / len(data['t'])

def cross_validation(data: tuple, num_folds: int, lambd_seq: np.ndarray):
    """This function takes training data, number of folds num_folds, and a
    sequence of lambdas lambd_seq as its arguments and returns the cross
    validation error across all lambdas.

    Inputs:
        - data: variable corresponding to the target vector (response) and 
                feature design matrix of the training dataset, structured as a 
                tuple.
                
        - num_folds: integer corresponding to the k number of folds used for
                     cross validation.
        
        - lambd_seq: a sequence of evenly spaced lambda values over a 
                     specified intereval.
                     
    Output:
        - a list of cross validation errors across all specified lambda. 
          Length of list is the same as length of lambd_seq.

    """
    data = shuffle_data(data)
    cv_error = np.zeros(len(lambd_seq))

    for i in range(len(lambd_seq)):
        lambd = lambd_seq[i]
        cv_loss_lmd = 0.0
        for fold in range(num_folds):
            val_cv, train_cv = split_data(data, num_folds, fold)
            model = train_model(train_cv, lambd)
            cv_loss_lmd += loss(val_cv, model)
        cv_error[i] = (cv_loss_lmd / num_folds)
        
    return cv_error

def training_test_errors(trainData, testData, lambd_seq):
    """This function takes the training set trainData and test set testData and
    returns the cross validation error across all lambdas for the adjusted model.
    
    Inputs:
        - trainData: training set
        
        - testData: test set
        
        - lambd_seq: a sequence of evenly spaced lambda values over a 
                     specified intereval.
    
    Outputs:
        - tuple of the training error and test error for each lambda
        
    """
    trainErrors = []
    testErrors = []
    
    for i in range(len(lambd_seq)):
        m = train_model(trainData, lambd_seq[i])
        trainE = loss(trainData, m)
        testE = loss(testData, m)
        trainErrors.append(trainE)
        testErrors.append(testE)
        
    return trainErrors, testErrors

# Generate data and lambda values:
data_train = {'X': np.genfromtxt('data_train_X.csv', delimiter=','),
              't': np.genfromtxt('data_train_y.csv', delimiter=',')}
data_test = {'X': np.genfromtxt('data_test_X.csv', delimiter=','),
             't': np.genfromtxt('data_test_y.csv', delimiter=',')}

lambdaList = np.arange(0.02, 1.5, 0.03)



# Calculating training and test error:
trainErr, testErr = find_errors(data_train, data_test, lambdaList)

# Calculating cv errors:
fiveFoldcv = cross_validation(data_train, 5, lambdaList)
tenFoldcv = cross_validation(data_train, 10, lambdaList)

fig, graph = plt.subplots()
graph.plot(lambdaList, trainErr, "o", label="training error")
graph.plot(lambdaList, testErr, "o", label="test error")
graph.plot(lambdaList, fiveFoldcv, "o", label="5 fold cv error")
graph.plot(lambdaList, tenFoldcv, "o", label="10 fold cv error")
graph.set(xlabel='lambda range', ylabel='errors',
       title='')
graph.grid()
graph.legend()
fig.savefig("q3.png")
plt.show()
