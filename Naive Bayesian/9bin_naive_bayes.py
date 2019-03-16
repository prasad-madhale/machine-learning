#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 19:58:24 2019

@author: prasad
"""
import pandas as pd
import numpy as np

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
def train_naive_bayes(train_data, train_labels, test_data, test_labels):
    
    mins = np.min(train_data, axis = 0)
    maxs = np.max(train_data, axis = 0)
    
    train_data = preprocess(train_data, mins, maxs)
    
    non_spam_indices = np.argwhere(train_labels == 0).flatten()
    spam_indices = np.argwhere(train_labels == 1).flatten()

    # get spam and non_spam data
    non_spam_data = np.take(train_data, non_spam_indices, axis = 0)
    spam_data = np.take(train_data, spam_indices, axis = 0)
    
    # priors
    priors = get_priors(train_labels)
        
    count_non_spam = count(non_spam_data)
    count_spam = count(spam_data)
    
    counts = np.array([count_non_spam, count_spam])    
    
    probabilities = prob(counts)
    
    predictions = []
    
    test_data = preprocess(test_data, mins, maxs)
    
    for pt in range(len(test_data)):
        data = test_data[pt]
        pred = (probabilities * data).sum(axis = 1) + priors
        predictions.append(np.argmax(pred))
        
    return acc(predictions, test_labels)
   
def get_priors(labels):
    count_spam = np.count_nonzero(labels)
    count_non_spam = len(labels) - count_spam
    priors = np.array([np.log(count_non_spam/ len(labels)), np.log(count_spam / len(labels))]) 
    return priors

def acc(preds, labels):
    check = (preds == labels).astype(int)
    count = np.count_nonzero(check)
    return count / len(labels)

def prob(feature_count):
    return np.log(feature_count / feature_count.sum(axis=1)[np.newaxis].T)
    
def count(data):
    greaters = np.sum(data, axis = 0) + 1.0
    return greaters

def preprocess(data, mins, maxs):
    
    for x in data:
        for f in range(len(x)):
            min_c = mins[f]
            max_c = maxs[f]
            
            # nine equal bins
            nine_bins = np.linspace(min_c, max_c, 9)
            
            x[f] = np.digitize(x[f], bins = nine_bins)-1
            
    return data

def get_labels(data):
    data_labels = data[:,-1]
    data = data[:,:-1]
    
    return data, data_labels

def train_folds(folds):    
    accs = []

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
                accuracy = train_naive_bayes(train_data, train_labels, test_data, test_labels)
                
                # each fold accuracies
                fold_acc.append(accuracy)
        
        # store mean of the accuracies obtained by using different folds as training set
        accs.append(np.mean(fold_acc))
    
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

# shuffle data
df = df.sample(frac = 1)

# create folds of the data based on customized conditions
folds = split(df)

# train naive bayes for provided folds
accs = train_folds(folds)

print('Accuracies for 10 Fold cross-validation')
print(accs)     
print('Mean Accuracy: {}'.format(np.mean(accs)))  