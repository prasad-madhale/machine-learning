#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 13:32:44 2019

@author: prasad
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

def get_data(names):
    '''
    Args
        column_names:  names of the features in dataset
    '''
    train = pd.read_csv('./data/train_wine.csv', sep=',',header=None)
    test = pd.read_csv('./data/test_wine.csv', sep=',',header=None)
    
    train.columns = names
    test.columns = names
    
    train_labels = train['label'].values
    test_labels = test['label'].values
    
    train = train.drop('label', axis=1).values
    test = test.drop('label', axis=1).values
    
    # normalize
    train = keras.utils.normalize(train, axis=1, order=2)
    test = keras.utils.normalize(test, axis=1, order=2)

    return train, test, train_labels, test_labels
    

class NN:
    def __init__(self, name: 'str', input_shape, out_shape, hidden_units, learn_rate = 0.01):
        self.name = name
        self.input_shape = input_shape
        self.out_shape = out_shape
        self.h_units = hidden_units
        self.lr = learn_rate
        self.initialize_weights()
        self.initialize_bias()
        
        
    def initialize_weights(self):
        self.w_ih = np.random.randn(self.input_shape[1], self.h_units) / np.sqrt(self.input_shape[1])
        self.w_ho = np.random.randn(self.h_units, self.out_shape[1]) / np.sqrt(self.h_units)

    def initialize_bias(self):
        self.b_ih = np.zeros((1, self.h_units))
        self.b_ho = np.zeros((1, self.out_shape[1]))
    
    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))
                
    def compute_grad(self, x):
        return x * (1.0 - x)
        
    def predict(self, inp):
        # input to hidden layer computations
        hidden = inp.dot(self.w_ih)
        hidden = np.add(hidden, self.b_ih)
        # input to hidden layer output
        a_ih = self.sigmoid(hidden)
        
        # hidden to output layer computations
        out = a_ih.dot(self.w_ho)
        out = np.add(out, self.b_ho)
        # hidden to output layer output
        a_ho = self.softmax(out)
            
        return np.argmax(a_ho, axis=1)
        
    def loss(self, pred_probs, labels):
        index = np.argmax(pred_probs, axis=1).astype(int)
        probs = pred_probs[np.arange(len(pred_probs)), index]
        logs = np.log(probs)
        return -np.sum(logs) / len(logs)
     
    def softmax(self, x):
        exps = np.exp(x)
        return exps / np.sum(exps, axis=1, keepdims= True)
    
    def train(self, inp, label, epochs):
        
        losses = []
        
        for epoch in range(epochs):
            # FORWARD PASS
            
            # input to hidden layer computations
            hidden = inp.dot(self.w_ih)
            hidden = np.add(hidden, self.b_ih)
    
            # input to hidden layer output
            a_ih = self.sigmoid(hidden)
            
            # hidden to output layer computations
            out = a_ih.dot(self.w_ho)
            out = np.add(out, self.b_ho)
            
            # hidden to output layer output
            a_ho = self.softmax(out)
           
            # LOSS CALCULATIONS
             
            # calculate loss at the end(output)
            loss = self.loss(a_ho, label)
            losses.append(loss)
            
            # BACK PROP
            
            # calculate delta w2
            grad_error_ho = (a_ho - label)
            deltaw2 = a_ih.T.dot(grad_error_ho)
            deltab2 = np.sum(grad_error_ho, axis=0)
            
            # calculate delta w1
            grad_error_ih = grad_error_ho.dot(self.w_ho.T) * self.compute_grad(a_ih)
            deltaw1 = inp.T.dot(grad_error_ih)
            deltab1 = np.sum(grad_error_ih, axis=0)
            
            self.w_ho -= self.lr * deltaw2
            self.w_ih -= self.lr * deltaw1
            self.b_ho -= self.lr * deltab2
            self.b_ih -= self.lr * deltab1
            
            if epoch % 10000 == 0:
                print('Epoch: {}, Loss: {}'.format(epoch, loss))
    
        return losses
    
def accuracy(predict, label):
    correct = 0
    label = np.argmax(label, axis=1)
    for x in range(len(predict)):
        if predict[x] == label[x]:
            correct += 1
            
    return correct / len(predict)
 
def plot_loss(loss):
    loss = moving_avg(loss)
    plt.figure(figsize = (20,10))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch (Cross Entropy Loss)')
    plt.plot(loss)

def moving_avg(values, window_size=1000):
    csum = np.cumsum(values)
    csum[window_size:] = csum[window_size:] - csum[:-window_size]
    return csum[window_size-1:] / window_size    


# HYPERPARAMETERS
NUM_CLASSES = 3
LEARNING_RATE = 0.0005
HIDDEN_UNITS = 16
EPOCHS = 200000

column_names = ['label', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13']
train, test, train_label, test_label = get_data(column_names)

# one hot encoding
test_label = keras.utils.to_categorical(test_label-1, NUM_CLASSES)
train_label = keras.utils.to_categorical(train_label-1, NUM_CLASSES)

np.random.seed(0)

nn = NN('Multi-Class Classifier', train.shape, train_label.shape, HIDDEN_UNITS, LEARNING_RATE)
loss = nn.train(train, train_label, EPOCHS)

plot_loss(loss)

pred_train = nn.predict(train)
print('Train Accuracy: {}'.format(accuracy(pred_train, train_label)))

pred_test = nn.predict(test)
print('Test Accuracy: {}'.format(accuracy(pred_test, test_label)))
