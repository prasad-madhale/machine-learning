#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 01:07:37 2019

@author: prasad
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def get_data(column_names):
    '''
    Args
        column_names: names of the features in dataset
    Returns
        train_df: training data
        test_df: testing data
    '''
    train_df = pd.read_csv('./data/housing_train.txt', delim_whitespace=True, header = None)
    test_df = pd.read_csv('./data/housing_test.txt', delim_whitespace=True, header = None)
    test_df.columns = column_names
    train_df.columns = column_names
    return train_df, test_df


def normalize(dataset):
    '''
    Args
        dataset: data to be normalized using shift-scale normalization
    Returns
        dataset: normalized dataset
        maxs: max parameters for each feature normalization
        mins: min parameters for each feature normalization
    '''
    maxs = dataset.max()
    mins = dataset.min()
    
    for feature in dataset.columns[:-1]:        
        for i, entry in dataset.iterrows():
            dataset.at[i, feature] = (entry[feature] - mins[feature]) / (maxs[feature] - mins[feature])
            
    return dataset, maxs, mins


def normalize_params(dataset, maxs, mins):
    '''
    Args
        dataset: data to be normalized
        maxs: max parameters for each feature normalization
        mins: min parameters for each feature normalization
    Returns:
        dataset: normalized dataset
    '''
    for feature in dataset.columns[:-1]:        
        for i, entry in dataset.iterrows():
            dataset.at[i, feature] = (entry[feature] - mins[feature]) / (maxs[feature] - mins[feature])
            
    return dataset


def predict(test_data, weights):
    '''
    Args
        test_data: data for which predictions are to be calculated
        weights: weights to obtain predictions based on
    Returns
        preds: predictions based on given weights applied on dataset
    '''
    test_data = test_data.drop(['MEDV'], axis = 1).values
    test_data = np.append(np.ones([len(test_data),1]),test_data,1)
 
    preds = {}
    
    for i in range(len(test_data)):
        preds[i] = np.dot(weights, test_data[i])
        
    return preds


def get_mse(test_data, preds):
    '''
    Args
        test_data: data for which model is to be tested using MSE
        preds: predictions on given test_data obtained from model
    Returns
        mse: mean squared error
    '''
    test_labels = test_data['MEDV'].values
    errors = []

    for i, label in enumerate(test_labels):
        errors.append(np.square(label - preds[i]))
    
    mse = pd.Series(errors).mean()
    return mse


def cost(data, labels, weights): 
    '''
    Args
        data: data for which cost needs to be calculated
        labels: actual labels for data used 
        weights: optimized weights for prediction
    Returns
        cost on the given data
    '''
    preds = np.dot(data, weights)
    preds = preds.flatten()
    
    return np.sum(np.square(np.subtract(preds, labels))) / len(data)
  
    
def train(train_data, learn_rate = 0.001, max_iter = 1000):
    '''
    Args
        train_data : normalized data for training
        learn_rate : learning rate for Gradient Descent
        max_iter : maximum number of iterations to run GD
    '''
    
    # get data without the labels
    x = train_data.drop(['MEDV'], axis = 1).values
    
    # add 1s to the data for bias calculations
    x = np.append(np.ones([len(x),1]),x,1)
    
    # get labels of the training set
    y = train_data['MEDV'].values
    
    # initialize weights with random values
    w = np.random.normal(scale = 1 / math.sqrt(len(x[0])),size = (len(x[0]), 1))
    
    w = w.flatten()
    
    # keep records of costs as we keep performing iteration of GD
    costs = []
    
    for itr in range(max_iter):   
        # predictions based on current weights
        predicts = np.dot(x, w)
        predicts = predicts.flatten()
        
        # difference between current predictions and actual labels
        loss = np.subtract(predicts, y)
        
        grads = np.dot(x.T, loss)
        
        # update weights
        w = np.subtract(w, learn_rate * grads)
        
        # record cost after weight updates
        costs.append(cost(x,y,w))
        
        if itr % 100 == 0:
            print('{}: Cost: {}'.format(itr, costs[itr]))
            
    return w, costs


def plot_cost(costs):
    plt.figure(figsize = (20,10))
    plt.title('Cost function')
    plt.ylabel('Costs')
    plt.xlabel('Iterations')
    plt.plot(costs)

#### EXECUTION

# names for the features
column_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

# extract data from files
train_data, test_data = get_data(column_names)

# normalize data
train_data, maxs, mins = normalize(train_data)

# normalize test data using same parameters as for the training set
test_data = normalize_params(test_data,maxs, mins)

# optimize weights using Gradient Descent  
w,costs = train(train_data)

# get predictions for optimized weights
pred_train = predict(train_data, w)

print('MSE for Housing dataset using Gradient Descent on Train Data: {}'.format(get_mse(train_data, pred_train)))

# get predictions for optimized weights
preds = predict(test_data, w)

print('MSE for Housing dataset using Gradient Descent on Test Data: {}'.format(get_mse(test_data, preds)))

plot_cost(costs)