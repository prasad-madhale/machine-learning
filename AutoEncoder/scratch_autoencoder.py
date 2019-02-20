#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 19:36:26 2019

@author: prasad
"""
import numpy as np
import matplotlib.pyplot as plt

def get_data():
    '''
    Args
        column_names:  names of the features in dataset
    '''
    data_frame = np.genfromtxt('./data/autoencoder_input.txt', delimiter = ',')
    return data_frame
    

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
        np.random.seed(0)
        self.w_ih = np.random.randn(self.input_shape[0], self.h_units) / np.sqrt(self.input_shape[0])
        self.w_ho = np.random.randn(self.h_units, self.out_shape[0]) / np.sqrt(self.h_units)

    def initialize_bias(self):
        self.b_ih = np.zeros((1, self.h_units))
        self.b_ho = np.zeros((1, self.out_shape[0]))
    
    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))
                
    def compute_grad(self, x):
        return x * (1-x)
        
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
        a_ho = self.sigmoid(out)
        
        return np.argmax(a_ho, axis=1)
        
    def loss(self, data, labels):
        square_diff = np.square(data - labels)
        return square_diff.mean()

    def train(self, inp, label, epochs):
        
        losses = []
        
        for epoch in range(epochs):
            # input to hidden layer computations
            hidden = inp.dot(self.w_ih)
            hidden = np.add(hidden, self.b_ih)
            # input to hidden layer output
            a_ih = self.sigmoid(hidden)
            
            # hidden to output layer computations
            out = a_ih.dot(self.w_ho)
            out = np.add(out, self.b_ho)
            # hidden to output layer output
            a_ho = self.sigmoid(out)
            
            # calculate loss at the end(output)
            loss = self.loss(a_ho, label)
            losses.append(loss)
            
            # calculate delta w2
            grad_error_ho = (a_ho - label) * self.compute_grad(a_ho) 
            deltaw2 = a_ih.T.dot(grad_error_ho)
            deltab2 = np.sum(grad_error_ho, axis=0)
            
            # calculate delta w1
            grad_error_ih = grad_error_ho.dot(self.w_ho.T) * self.compute_grad(a_ih)
            deltaw1 = inp.T.dot(grad_error_ih)
            deltab1 = np.sum(grad_error_ih, axis=0)
            
            self.w_ho -= deltaw2 * self.lr
            self.w_ih -= deltaw1 * self.lr
            self.b_ho -= deltab2 * self.lr
            self.b_ih -= deltab1 * self.lr
            
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
    plt.figure(figsize = (20,10))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch (Autoencoder)')
    plt.plot(loss)

def moving_avg(values, window_size=1000):
    csum = np.cumsum(values)
    csum[window_size:] = csum[window_size:] - csum[:-window_size]
    return csum[window_size-1:] / window_size


# HYPERPARAMETERS
HIDDEN_UNITS = 3
LEARNING_RATE = 0.09
EPOCHS = 3000

inp = get_data()
out = inp
nn = NN('AutoEncoder', inp.shape, out.shape, HIDDEN_UNITS, LEARNING_RATE)
loss = nn.train(inp, out, EPOCHS)

plot_loss(loss)

predictions = nn.predict(inp)

print('Accuracy: {}'.format(accuracy(predictions, out)))