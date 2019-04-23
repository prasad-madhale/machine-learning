#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: prasad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import PCA


def get_data():
    test_data = pd.read_csv('./data/spam_polluted/test_feature.txt', sep=' ')
    test_label = pd.read_csv('./data/spam_polluted/test_label.txt', sep='\n')
    train_data = pd.read_csv('./data/spam_polluted/train_feature.txt', sep=' ')
    train_label = pd.read_csv('./data/spam_polluted/train_label.txt', sep='\n')

    # flatten labels
    test_label = test_label.values.flatten()
    train_label = train_label.values.flatten()

    return train_data, train_label, test_data, test_label


def split(data, num_of_splits=10):
    # Special Splitting
    # Group 1 will consist of points {1,11,21,...}, Group 2 will consist of 
    # points {2,12,22,...}, ..., and Group 10 will consist of points {10,20,30,...}
    
    folds = []
    
    for k in range(num_of_splits):
        fold = []
        
        for i in range(k, len(data), num_of_splits):
            fold.append(data[i])
        
        fold = np.array(fold)
        folds.append(fold)
        
    return folds


# assuming features are bernoulli variables
def train_naive_bayes(train_data, train_labels, test_data, test_labels, error_table, preds, labels):
    
    non_spam_indices = np.argwhere(train_labels == 0).flatten()
    spam_indices = np.argwhere(train_labels == 1).flatten()

    # get spam and non_spam data
    non_spam_data = np.take(train_data, non_spam_indices, axis=0)
    spam_data = np.take(train_data, spam_indices, axis=0)
    
    data = np.array([non_spam_data, spam_data])
    
    models = mean_var(data)
    
    probs = get_prob(models, test_data)
    
    pred = np.argmax(probs, axis=1)

    acc = sum(pred == test_labels) / len(test_labels)
    
    c_mat = conf_matrix(pred, test_labels)
    error_table.append(c_mat)
    preds.append(pred)
    labels.append(test_labels)
    
    return acc


def get_priors(labels):
    count_spam = np.count_nonzero(labels)
    count_non_spam = len(labels) - count_spam
    priors = np.array([np.log10(count_non_spam/ len(labels)), np.log10(count_spam / len(labels))]) 
    return priors


def acc(preds, labels):
    check = (preds == labels).astype(int)
    count = np.count_nonzero(check)
        
    return count / len(labels)


def gauss(x, mean, std):
    
    if std == 0:
        std = 1e-4
        
    exp_term = np.exp(- ((x - mean)**2 / (2 * std**2)))
    denom = (np.sqrt(2 * np.pi) * std)
    
    prob = (exp_term / denom)
    
    return np.log10(prob)


def get_prob(model, data):
    final_probs = []
    
    for x in data:
        probs = []
        
        for m in model:                
            p = sum(gauss(i, *s) for s, i in zip(m, x))
            probs.append(p)
        
        final_probs.append(probs)
       
    return final_probs
        

def mean_var(data):
    
    model = []
    
    for data_class in data:
        mean = np.mean(data_class, axis = 0)
        var = np.std(data_class, axis = 0)
        
        model.append(np.c_[mean, var])
    
    return np.array(model)


def binarize(data, mean):
    data = (data < mean).astype(int)
    return data


def get_labels(data):
    data_labels = data[:, -1]
    data = data[:, :-1]
    
    return data, data_labels


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
    
    return np.array([tp, fp, fn, tn])


def plot_roc(truth, preds):
    fprs, tprs, _ = metrics.roc_curve(truth, preds)
    
    plt.figure(figsize = (15,10))
    plt.title('ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    plt.plot(fprs, tprs, label = 'AUC: {}'.format(metrics.auc(fprs, tprs)))
    plt.legend(loc = 'lower right')


def normalize(data):
    # return preprocessing.minmax_scale(data, feature_range=(0, 1))

    std = preprocessing.StandardScaler()
    std.fit(data)
    std.transform(data)

    return data


def train_folds(folds):
    
    accs = []
    error_table = []
    predictions = []
    labels = []
    
    # for each fold
    for k in range(len(folds)):  
        # kth fold selected as test set
        test_data = np.array(folds[k])
        test_data, test_labels = get_labels(test_data)
        
        fold_acc = []        
        
        # all other folds used for training
        for j in range(len(folds)):
            
            if j != k:
                train_data = np.array(folds[j])
                train_data, train_labels = get_labels(train_data)
                accuracy = train_naive_bayes(train_data, train_labels, test_data, test_labels, error_table, predictions, labels)

                # each fold accuracies
                fold_acc.append(accuracy)
        
        # store mean of the accuracies obtained by using different folds as training set
        accs.append(np.mean(fold_acc))
    
    mean_errors = np.mean(error_table, axis=0)
    error_table.append(mean_errors)
    
    max_idx = np.argmax(accs)

    print('Error Tables')
    print('TP    FP  FN    TN')
    print(*error_table, sep='\n')

    plot_roc(predictions[max_idx], labels[max_idx])
    
    return accs


def preprocess(data, train_label, test_label):

    # normalize data
    data = normalize(data)

    # combine labels as well
    labels = np.concatenate([train_label, test_label])

    # reshape labels to combine
    labels = labels.reshape((labels.shape[0], 1))

    # combine labels with actual data
    data = np.hstack((data, labels))

    return data


def pca(train_data, test_data):

    data = np.concatenate([train_data, test_data])

    # reduce features to 100 which are linear combination of each other
    pca = PCA(n_components=100)

    data = pca.fit_transform(data)

    return data


with open('./logs/out.txt', 'w') as f:
    # get data from txt file
    train_data, train_label, test_data, test_label = get_data()

    data = pca(train_data, test_data)

    # add labels to the data
    data = preprocess(data, train_label, test_label)

    # create folds of the data based on customized conditions
    folds = split(data)

    # # train naive bayes for provided folds
    accs = train_folds(folds)

    print(accs, file=f)
    print('Mean Accuracy for 10 Fold cross-validation', file=f)
    print(np.mean(accs), file=f)

print('Done Training!')
