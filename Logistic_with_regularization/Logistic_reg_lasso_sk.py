#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 01:07:37 2019

@author: prasad
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score


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


def train(train_data, train_labels, alpha=0.0002):
    '''
    Args
        train_data : normalized data for training
        learn_rate : learning rate for Gradient Descent
        max_iter : maximum number of iterations to run GD
    '''
    model = Lasso(alpha=alpha)
    model.fit(train_data, train_labels)

    return model


def thresholds_checker(m, train_data, train_labels):

    train_preds = m.predict(train_data)
    rounds = np.vectorize(rounder)

    ts = np.linspace(0, 1, 500)

    accs = {}

    for t in ts:
        preds = rounds(train_preds, t)
        accs[t] = accuracy_score(train_labels, preds)

    return max(accs, key=accs.get)


def get_accuracy(model, data, label, threshold):
    pred = model.predict(data)
    rounds = np.vectorize(rounder)

    pred = rounds(pred, threshold)

    return accuracy_score(label, pred)


# EXECUTION
# extract data from files
train_data, train_labels, test_data, test_labels = get_data()

# normalize the data
train_data, test_data = normalize(train_data, test_data)

# optimize weights using Gradient Descent 
model = train(train_data, train_labels)

best_threshold = thresholds_checker(model, train_data, train_labels)

train_acc = get_accuracy(model, train_data, train_labels, best_threshold)
test_acc = get_accuracy(model, test_data, test_labels, best_threshold)

print('Train acc: {}, Test acc: {}'.format(train_acc, test_acc))
