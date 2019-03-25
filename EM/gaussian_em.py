#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 20:29:49 2019

@author: prasad
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from collections import Counter

def get_data():
    df = pd.read_csv('./data/2gaussian.txt', sep = ' ')
    return df

def guess(data, param, mu, sig):
    p = param
    
    for i in range(len(data)):
        p *= norm.pdf(data[i], mu[i], sig[i][i])
        
    return p
    
def e_step(model, data):
    
    y = np.empty(data.shape[0])
    
    for i in range(len(data)):
        
        p_gaussian1 = guess(data[i], model['lambda'][0], model['mu1'], model['sig1'])
        p_gaussian2 = guess(data[i], model['lambda'][1], model['mu2'], model['sig2'])
        
        if p_gaussian1 > p_gaussian2:
            y[i] = 1
        else:
            y[i] = 2
        
    return y
    
def maximization(model, data, labels, count):
    # update model parameters using new labels and counts
    new_model = update_model(model.copy(), data, labels, count)
    
    # calculates the error between old and new model parameters
    err = error(model, new_model)
    
    return err, new_model
    
def error(old_model, new_model):
    
    mus = ['mu1', 'mu2']
    err = 0
    
    for mu in mus:
        for i in range(len(mus)):
            err += (old_model[mu][i] - new_model[mu][i])**2         
    
    return err**(1/2)
    
def update_model(model, data, labels, count):
    # get indices of all the datapoints for each gaussian
    g1_indices = np.argwhere(labels == 1).flatten()
    g2_indices = np.argwhere(labels == 2).flatten()
    
    # get the data for each gaussian using provided datapoints
    gaussian1_data = np.take(data, g1_indices, axis = 0)
    gaussian2_data = np.take(data, g2_indices, axis = 0)
    
    # calculate gaussian probabilities
    g1_percent = len(gaussian1_data) / len(data)
    g2_percent = len(gaussian2_data) / len(data)

    # perform the updates to model
    model['lambda'] = [g1_percent, g2_percent]
    model['mu1'] = np.mean(gaussian1_data, axis=0)
    model['mu2'] = np.mean(gaussian2_data, axis=0)
    
    g1_std = np.std(gaussian1_data, axis=0)
    g2_std = np.std(gaussian2_data, axis=0)
    
    model['sig1'] = [[g1_std[0],0],[0,g1_std[1]]]
    model['sig2'] = [[g2_std[0],0],[0,g2_std[1]]] 
    
    print('Number of points in gaussians: {},{}'.format(len(gaussian1_data), len(gaussian2_data)))
    
    # return new model
    return model
    
   
threshold = 1e-4
df = get_data()
vals = df.values
err = float('inf')
itr = 0

# random initialization of parameters
model = {'mu1': [1,1], 'mu2': [10,10], 'sig1': [[1,0],[0,1]], 'sig2': [[1,0],[0,1]], 'lambda': [0.5, 0.5]}

# stop if we converge
while err > threshold: 
    itr += 1
    
    # Step 1: 
    # Expectation > Assign each data point a corresponding gaussian based on the current
    #               parameters
    labels = e_step(model, df.values)
    
    count = Counter(labels)
    
    # Step 2:
    # Maximization > Update the model parameters based on new labels and calculate the error
    err, new_model = maximization(model, df.values, labels, count)
    
    print('Iteration: {}, Error: {}'.format(itr, err))
    
    # assign updated model to the current model in order to continue
    # EM process
    model = new_model

print(model)
    