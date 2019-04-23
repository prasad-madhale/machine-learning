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
from sklearn import metrics
from sklearn import preprocessing


def get_data():
    test_data = pd.read_csv('./data/spam_polluted/test_feature.txt', sep=' ')
    test_label = pd.read_csv('./data/spam_polluted/test_label.txt', sep='\n')
    train_data = pd.read_csv('./data/spam_polluted/train_feature.txt', sep=' ')
    train_label = pd.read_csv('./data/spam_polluted/train_label.txt', sep='\n')

    # flatten labels
    test_label = test_label.values.flatten()
    train_label = train_label.values.flatten()

    return train_data, train_label, test_data, test_label


def normalize(train_data, test_data):

    # combine test and train data for normalization
    data = np.concatenate([train_data, test_data])

    # normalize
    data = preprocessing.minmax_scale(data, feature_range=(0, 1))

    train_data = data[:len(train_data)]
    test_data = data[len(train_data):]

    return train_data, test_data


def predict(test_data, test_labels, weights, threshold=0.45):
    test_data = np.append(np.ones([len(test_data), 1]), test_data, 1)
    
    preds = {}
    
    for i in range(len(test_data)):
        preds[i] = rounder(sigmoid(np.dot(test_data[i], weights)), threshold)
    
    conf, tp, fp, tn, fn = conf_matrix(preds, test_labels)
    return preds, conf, tp, fp, tn, fn


def rounder(x, threshold):
    '''
    Args
        x: exact prediction
    Returns
        label based on the threshold value
    '''
    if x >= threshold:
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


def accuracy(test_data, test_labels, preds):

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


def train(train_data, train_labels, reg_strength=0.1, learn_rate=0.001, max_iter=3000):
    '''
    Args
        train_data : normalized data for training
        learn_rate : learning rate for Gradient Descent
        max_iter : maximum number of iterations to run GD
    '''
    print('Learning rate: {}'.format(learn_rate))
    print('Max Iterations: {}'.format(max_iter))

    # add 1s to the data for bias calculations
    train_data = np.append(np.ones([len(train_data), 1]), train_data, 1)

    # initialize weights with random values
    w = np.random.normal(scale=1 / math.sqrt(len(train_data[0])), size=(len(train_data[0]), 1))
    
    # keep records of costs as we keep performing iteration of GD
    loglikes = []
    
    w = w.flatten()
        
    for itr in range(max_iter+1):           
        preds = np.dot(train_data, w)
        sigs = np.vectorize(sigmoid)
        preds = sigs(preds.flatten())
        
        loss = np.subtract(train_labels, preds)

        # regularization term
        grads = np.dot(train_data.T, loss) + ((reg_strength / train_data.shape[0]) * w)

        w = np.add(w, learn_rate * grads)
        
        # record cost after weight updates
        loglikes.append(log_likelihood(train_data, train_labels, w))
        
        if itr % 500 == 0:
            print('{}: Log-Likelihood: {}'.format(itr, loglikes[itr]))
            
    return w, loglikes


def plot_likelihood(logs):
    plt.figure(figsize=(15,10))
    plt.title('Log Likelihood')
    plt.xlabel('Iterations')
    plt.plot(logs)


def conf_matrix(preds, test_labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for i in range(len(preds)):
        p = preds[i]
        if p == test_labels[i]:
            if p == 1:
                tp += 1
            else:
                tn += 1
        else:
            if p == 1:
                fp += 1
            else:
                fn += 1
    
    conf = np.array([[tp, fp], [fn, tn]])
    
    return conf, tp, fp, tn, fn


def roc(test_data, test_labels, w):
    ts = np.arange(0,1, 0.01)
    tprs = []
    fprs = []
    
    for t in ts:
        _, conf, tp, fp, tn, fn = predict(test_data, test_labels, w, t)
        tpr = tp/(tp + fn)
        fpr = fp/(fp + tn)
        tprs.append(tpr)
        fprs.append(fpr)
    
    plot_roc(fprs, tprs)
    
    
def plot_roc(fprs, tprs):
    plt.figure(figsize = (15, 10))
    plt.title('ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    plt.plot(fprs, tprs, label = 'AUC: {}'.format(metrics.auc(fprs, tprs)))
    plt.legend(loc = 'lower right')
    plt.show()

    
# EXECUTION

with open('./logs/out_ridge', 'w') as file_op:
    # extract data from files
    train_data, train_labels, test_data, test_labels = get_data()

    # normalize the data
    train_data, test_data = normalize(train_data, test_data)

    # optimize weights using Gradient Descent
    w, loglike = train(train_data, train_labels)

    # get predictions for training data
    pred_train, _, _, _, _, _ = predict(train_data, train_labels, w)

    # get accuracy percentage of the predictions
    train_acc = accuracy(train_data, train_labels, pred_train)
    print('Logistic Regression using Gradient Descent Training Accuracy on SpamBase: {}'.format(train_acc), file=file_op)

    # get predictions for optimized weights
    preds, conf, _, _, _, _ = predict(test_data, test_labels, w)

    # get accuracy percentage of the predictions
    acc = accuracy(test_data, test_labels, preds)
    print('Logistic Regression using Gradient Descent Testing Accuracy on SpamBase: {}'.format(acc), file=file_op)

    print('Confusion Matrix:', file=file_op)
    print(conf, file=file_op)

    plot_likelihood(loglike)

    roc(test_data, test_labels, w)
