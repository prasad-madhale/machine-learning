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
        column_names:  names of the features in dataset
    '''
    data_frame = pd.read_csv('./data/spambase.txt', sep = ',')
    data_frame.columns = column_names
    
    return data_frame


def normalize(dataset):
    '''
    Args
        dataset: data to be normalized using shift-scale normalization
    Returns
        dataset: normalized dataset
    '''
    maxs = dataset.max()
    mins = dataset.min()
    
    for feature in dataset.columns[:-1]:        
        for i, entry in dataset.iterrows():
            dataset.at[i, feature] = (entry[feature] - mins[feature]) / (maxs[feature] - mins[feature])
            
    return dataset


def train_test_split(dataframe, percent = 20):
    '''
    Args
        dataframe: dataset for the problem
        percent: percentage of total dataframe to be used as test data
    Returns
        train_data: training data
        test_data: testing data
    '''
    test_data_size = int(len(dataframe) * percent / 100)
    test_data = dataframe[:test_data_size]    
    train_data = dataframe[test_data_size:]
    
    return train_data, test_data


def predict(test_data, weights):
    '''
    Args
        test_data: data for which predictions are to be calculated
        weights: weights to obtain predictions based on
    Returns
        preds: predictions based on given weights applied on dataset
    '''
    test_data = test_data.drop(['spam_label'], axis = 1).values
    test_data = np.append(np.ones([len(test_data),1]),test_data,1)
 
    preds = {}
    
    for i in range(len(test_data)):
        preds[i] = rounder(sigmoid(np.dot(test_data[i], weights)))
        
    return preds


def rounder(x):
    '''
    Args
        x: exact prediction
    Returns
        label based on the threshold value
    '''
    if x >= 0.26:
        return 1
    return 0


def get_mse(test_data, preds):
    '''
    Args
        test_data: data for which model is to be tested using MSE
        preds: predictions on given test_data obtained from model
    Returns
        mse: mean squared error
    '''
    test_labels = test_data['spam_label'].values
    errors = []

    for i, label in enumerate(test_labels):
        errors.append(np.square(label - preds[i]))
    
    mse = pd.Series(errors).mean()
    return mse


def accuracy(test_data, preds):
    test_labels = test_data['spam_label'].values
    
    correct_count = 0
    
    for i in range(len(preds)):
        if test_labels[i] == preds[i]:
            correct_count += 1
        
    return correct_count / len(test_labels)


def log_likelihood(data, labels, weights): 
    '''
    Args
        data: data for which cost needs to be calculated
        labels: actual labels for data used 
        weights: optimized weights for prediction
    Returns
        cost on the given data
    '''
    predictions = np.dot(data, weights)
    predictions = predictions.flatten()
    
    return np.sum(labels*predictions - np.log(1 + np.exp(predictions)))


def sigmoid(z):
    return 1/ (1+np.exp(-z))


def train(train_data, learn_rate = 0.001, max_iter = 1500):
    '''
    Args
        train_data : normalized data for training
        learn_rate : learning rate for Gradient Descent
        max_iter : maximum number of iterations to run GD
    '''
    print('Learning rate: {}'.format(learn_rate))
    print('Iterations: {}'.format(max_iter))
    
    # get data without the labels
    x = train_data.drop(['spam_label'], axis = 1).values
    
    # add 1s to the data for bias calculations
    x = np.append(np.ones([len(x),1]),x,1)
    
    # get labels of the training set
    y = train_data['spam_label'].values
    
    # initialize weights with random values
    w = np.random.normal(scale = 1 / math.sqrt(len(x[0])),size = (len(x[0]), 1))
    
    # keep records of costs as we keep performing iteration of GD
    loglikes = []
    
    w = w.flatten()
        
    for itr in range(max_iter+1):           
        preds = np.dot(x, w)
        sigs = np.vectorize(sigmoid)
        preds = sigs(preds.flatten())
        
        loss = np.subtract(y, preds)
        
        grads = np.dot(x.T, loss)
        
        w = np.add(w, learn_rate * grads)
        
        # record cost after weight updates
        loglikes.append(log_likelihood(x,y,w))
        
        if itr % 100 == 0:
            print('{}: Log-Likelihood: {}'.format(itr, loglikes[itr]))
            
    return w, loglikes

def plot_likelihood(logs):
    plt.figure(figsize = (20,10))
    plt.title('Log Likelihood')
    plt.xlabel('Iterations')
    plt.plot(logs)
    
### EXECUTION
    
# names for the features
column_names = ['word_freq_make','word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our', 
               'word_freq_over','word_freq_remove','word_freq_internet','word_freq_order','word_freq_mail',
               'word_freq_receive','word_freq_will','word_freq_people','word_freq_report','word_freq_addresses',
               'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you',
               'word_freq_credit','word_freq_your','word_freq_font','word_freq_000','word_freq_money',
               'word_freq_hp','word_freq_hpl','word_freq_george','word_freq_650','word_freq_lab',
               'word_freq_labs','word_freq_telnet','word_freq_857','word_freq_data','word_freq_415',
               'word_freq_85','word_freq_technology','word_freq_1999','word_freq_parts','word_freq_pm',
               'word_freq_direct','word_freq_cs','word_freq_meeting','word_freq_original','word_freq_project',
               'word_freq_re','word_freq_edu','word_freq_table','word_freq_conference','char_freq_;',
               'char_freq_(','char_freq_[','char_freq_!','char_freq_$','char_freq_#','capital_run_length_average',
               'capital_run_length_longest','capital_run_length_total','spam_label']
    
# extract data from files
dataframe = get_data(column_names)

# normalize the data
dataframe = normalize(dataframe)    

# split data into test and training data
train_data, test_data = train_test_split(dataframe)

# optimize weights using Gradient Descent 
w, loglike = train(train_data)

## get predictions for training data
pred_train = predict(train_data, w)
#print('MSE for SpamBase using Gradient Descent on Train Data: {}'.format(get_mse(train_data, pred_train)))

# get accuracy percentage of the predictions
train_acc = accuracy(train_data, pred_train)
print('Accuracy for SpamBase using Gradient Descent on Train Data: {}'.format(train_acc))

## get predictions for optimized weights
preds = predict(test_data, w)
#print('MSE for SpamBase using Gradient Descent on Test Data: {}'.format(get_mse(test_data, preds)))

# get accuracy percentage of the predictions
acc = accuracy(test_data, preds)
print('Accuracy for SpamBase using Gradient Descent on Test Data: {}'.format(acc))

plot_likelihood(loglike)