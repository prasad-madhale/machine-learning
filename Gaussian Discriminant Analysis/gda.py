#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 19:58:13 2019

@author: prasad
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def get_data(column_names):
    data_frame = pd.read_csv('./data/spambase.txt', header = None, sep = ',')
    data_frame.columns = column_names
    return data_frame

def k_fold(dataframe, num_folds = 10):
    folds = KFold(num_folds, random_state = 2, shuffle = True)
    return folds

def split_data_labels(train_data, test_data):
    # get labels for test and train data
    test_labels = test_data['spam_label'].values
    train_labels = train_data['spam_label'].values

    # drop the labels from test and train data
    train_data = train_data.drop(['spam_label'], axis = 1).values
    test_data = test_data.drop(['spam_label'], axis = 1).values

    return train_data, test_data, train_labels, test_labels

def accuracy(probs0, probs1, test_labels):
    predict = np.sign(np.subtract(probs1, probs0))
    predict[predict < 0] = 0

    count = 0
    for i in range(len(test_labels)):
        if test_labels[i] == predict[i]:
            count += 1

    return count / len(test_labels)

def train_gda(train_data, test_data, train_labels, test_labels):
    non_spam_indices = np.argwhere(train_labels == 0).flatten()
    spam_indices = np.argwhere(train_labels == 1).flatten()

    # get spam and non_spam data
    non_spam_data = np.take(train_data, non_spam_indices, axis = 0)
    spam_data = np.take(train_data, spam_indices, axis = 0)

    # get number of spam and non-spam data points in training data
    num_non_spam = len(non_spam_data) / len(train_data)
    num_spam = len(spam_data) / len(train_data)

    # means for spam and non-spam data
    non_spam_mean = np.mean(non_spam_data, axis = 0)
    spam_mean = np.mean(spam_data, axis = 0)

    # covariance matrix
    cov = np.cov(train_data.T)

    n = len(train_data[0])

    probs0 = list()
    probs1 = list()

    for x in range(len(test_data)):
        diff_non_spam = (test_data[x] - non_spam_mean)
        diff_spam = (test_data[x] - spam_mean)

        const = 1 / (((2 * np.pi)**(n/2)) * np.linalg.det(cov)**0.5)

        # build probabilities
        prob0 = const * (np.exp(-0.5 * np.dot(diff_non_spam, np.linalg.inv(cov)).dot(diff_non_spam.T)))
        prob1 = const * (np.exp(-0.5 * np.dot(diff_spam, np.linalg.inv(cov)).dot(diff_spam.T)))

        probs0.append(prob0 * num_non_spam)
        probs1.append(prob1 * num_spam)

    return probs0, probs1


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

df = get_data(column_names)
folds = k_fold(df)

accuracies = []

for train,test in folds.split(df):
    train_data = df.iloc[train]
    test_data = df.iloc[test]

    train_data, test_data, train_labels, test_labels = split_data_labels(train_data, test_data)
    prob0, prob1 = train_gda(train_data, test_data, train_labels, test_labels)

    acc = accuracy(prob0, prob1, test_labels)
    accuracies.append(acc)

print('Accuracies over 10 folds and shuffled data')
print(accuracies)
mean_acc = np.mean(accuracies)
print('Mean Accuracy: {}'.format(mean_acc))
