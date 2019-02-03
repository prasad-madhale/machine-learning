#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 03:40:20 2019

@author: prasad
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def get_data(column_names):
    df = pd.read_csv('./data/perceptronData.txt', delim_whitespace=True, header = None)
    df.columns = column_names
    return df


def perceptron(train_data, learn_rate = 0.0001, max_iter = 1000):
    print('Learning rate: {}'.format(learn_rate))
    print('Max Iterations: {}'.format(max_iter))
    
    # get data without the labels
    x = train_data.drop(['labels'], axis = 1).values
    
    # add 1s to the data for bias calculations
    x = np.append(np.ones([len(x),1]),x,1)
    
    # initialize weights with random values
    w = np.random.normal(scale = 1 / math.sqrt(len(x[0])),size = (len(x[0]), 1))
    
    w = w.flatten()
    
    mistakes = []
    
    for itr in range(max_iter):
        
        preds = np.dot(x,w)
        preds = preds.flatten()
        
        mistake = [x[i] for i,pred in enumerate(preds) if pred < 0]
        
        for entry in mistake:
            w = np.add(w, learn_rate * entry.T)
        
        print('Iteration: {}, total_mistake: {}'.format(itr, len(mistake)))
        
        mistakes.append(len(mistake))
        
        if len(mistake) == 0:
            break
    
    normalized_w = normalize_wt(w)
    return w, normalized_w, mistakes 
    

def plot_mistakes(mistakes):
    plt.figure(figsize = (15,10))
    plt.title('Mistakes over iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Number of Mistakes')
    plt.plot(mistakes)
    

def preprocess(data):
    data.loc[data['labels'] < 0, data.columns] *= -1 
    return data


def normalize_wt(w):
    w0 = -w[0]
    w = w[1:]
    
    return w / w0

        
column_names = ['feature1', 'feature2', 'feature3', 'feature4', 'labels']
dataframe = get_data(column_names)
dataframe = preprocess(dataframe)
w, norm_w, mistakes = perceptron(dataframe)
print('Classifier weights: {}'.format(w))
print('Normalized with threshold: {}'.format(norm_w))
plot_mistakes(mistakes)
