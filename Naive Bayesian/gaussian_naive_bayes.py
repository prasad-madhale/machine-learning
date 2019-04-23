#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 23:00:46 2019

@author: prasad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def get_data(column_names):
    data_frame = pd.read_csv('./data/spambase.txt', header = None, sep = ',')
    data_frame.columns = column_names
    return data_frame

def split(data, num_of_splits = 10):
    # Special Splitting
    # Group 1 will consist of points {1,11,21,...}, Group 2 will consist of
    # points {2,12,22,...}, ..., and Group 10 will consist of points {10,20,30,...}

    folds = []

    for k in range(num_of_splits):
        fold = []

        for i in range(k,len(data),num_of_splits):
            fold.append(data.iloc[i])

        fold = np.array(fold)
        folds.append(fold)

    return folds

# assuming features are bernoulli variables
def train_naive_bayes(train_data, train_labels, test_data, test_labels, error_table, preds, labels):

    non_spam_indices = np.argwhere(train_labels == 0).flatten()
    spam_indices = np.argwhere(train_labels == 1).flatten()

    # get spam and non_spam data
    non_spam_data = np.take(train_data, non_spam_indices, axis = 0)
    spam_data = np.take(train_data, spam_indices, axis = 0)

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
            p = sum(gauss(i, *s) for s, i in zip(m ,x))
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
    data_labels = data[:,-1]
    data = data[:,:-1]

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

    mean_errors = np.mean(error_table, axis = 0)
    error_table.append(mean_errors)

    max_idx = np.argmax(accs)

    print('Error Tables')
    print('TP    FP  FN    TN')
    print(*error_table, sep='\n')

    plot_roc(predictions[max_idx], labels[max_idx])

    return accs


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

# get data from txt file
df = get_data(column_names)

df = df.sample(frac=1)

df = normalize(df)

# create folds of the data based on customized conditions
folds = split(df)

# train naive bayes for provided folds
accs = train_folds(folds)

print(accs)
print('Mean Accuracy for 10 Fold cross-validation')
print(np.mean(accs))
