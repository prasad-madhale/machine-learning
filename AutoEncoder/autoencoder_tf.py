#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 02:38:44 2019

@author: prasad
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras


def get_data():
    '''
    Args
        column_names:  names of the features in dataset
    '''
    data_frame = np.genfromtxt('./data/autoencoder_input.txt', delimiter = ',', dtype='float32')
    return data_frame


class NN:
    def __init__(self, name: 'str', inp_shape, out_shape, hid_unit, learn_rate, sess):
        self.scope_name = name
        self.sess = sess
        self.inp_shape = inp_shape
        self.out_shape = out_shape
        self.lr = learn_rate
        self.h_unit = hid_unit
        
        with tf.variable_scope('inputs'):
            self.input = tf.placeholder(shape=[None, self.inp_shape], name='inputs', dtype=tf.float32)
            self.labels = tf.placeholder(shape=[None,self.out_shape], name='labels', dtype=tf.float32)
        
        self.predict = self.build_network()
        
        with tf.variable_scope('loss'):
            self.loss = tf.losses.mean_squared_error(self.labels,self.predict)
            self.loss_plot = tf.summary.scalar('loss', self.loss)
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        
        self.summaries = tf.summary.merge_all()
        
        with tf.variable_scope('accuracy'):
            _, self.acc = tf.metrics.accuracy(tf.math.argmax(self.labels,1), 
                                              tf.math.argmax(self.predict,1))
        
    def build_network(self):
        reg = keras.regularizers.l2(0.1)
        lyr1 = tf.layers.dense(self.input, activation=tf.nn.sigmoid, units=self.h_unit,
                                kernel_initializer = tf.initializers.glorot_uniform(seed=0),
                                kernel_regularizer = reg)
        out = tf.layers.dense(lyr1, activation=tf.nn.sigmoid, units=self.out_shape)
        return out

    def train(self, data, label):                
        loss, _, acc = self.sess.run([self.loss, self.train_op, self.acc],
                                     feed_dict={self.input: data,
                                                self.labels: label})
        return loss, acc
    
    def get_summary(self, data, labels):
        return self.sess.run(self.summaries, feed_dict = {self.input: data,
                                                          self.labels: labels})
    
    def test(self, data, labels):
        return self.sess.run(self.acc, feed_dict={self.input: data, self.labels: labels})
    

# HYPERPARAMETERS
HIDDEN_UNITS = 3
LEARNING_RATE = 0.05
EPOCHS = 10000
  
inp = get_data()
label = inp

tf.reset_default_graph()

with tf.Session() as sess:
    nn = NN('AutoEncoder', inp.shape[1], label.shape[1], HIDDEN_UNITS, LEARNING_RATE, sess)
    tensor_plot = tf.summary.FileWriter('log/AutoEncoder', graph = sess.graph)
    
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(EPOCHS):
        loss, acc = nn.train(inp, label)
                
        if epoch % 1000 == 0:
            print('Epoch: {}, Loss: {}, Accuracy: {}'.format(epoch, loss, acc))
        
        summary = nn.get_summary(inp, label)
        tensor_plot.add_summary(summary, epoch)            
        