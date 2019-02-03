#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 03:40:20 2019

@author: prasad
"""

import pandas as pd


def get_data(column_names):
    df = pd.read_csv('./data/perceptronData.txt', delim_whitespace=True, header = None)
    df.columns = column_names
    return df


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
    
column_names = ['feature1', 'feature2', 'feature3', 'feature4', 'label']
dataframe = get_data(column_names)
dataframe,_,_ = normalize(dataframe)
