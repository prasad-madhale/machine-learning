#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 06:35:13 2019

@author: prasad
"""

import pandas as pd
import tensorflow as tf
from tensorflow import keras

def get_data(names):
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


## TRAIN MODELS(Tensorflow)

class NN:
    def __init__(self, name: str, sess, in_shape, out_shape, lr = 0.001):
        self.scope_name = name
        self.sess = sess
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.lr = lr
        self.input = tf.placeholder(shape=[None, self.in_shape], name='inputs', dtype=tf.float32)
        self.labels = tf.placeholder(shape=[None,self.out_shape], name='labels', dtype=tf.float32)
        
        self.predict = self.build_network()
        
        with tf.variable_scope('loss'):
            self.loss = tf.losses.softmax_cross_entropy(self.labels, self.predict)
            self.loss_plot = tf.summary.scalar('loss', self.loss)
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        
        self.summaries = tf.summary.merge_all()
        
        with tf.variable_scope('accuracy'):
            _, self.acc = tf.metrics.accuracy(tf.math.argmax(self.labels,1),
                                              tf.math.argmax(self.predict,1))
        
    def build_network(self):
        with tf.variable_scope('mlp'):
            reg = keras.regularizers.l2(0.1)
            lyr1 = tf.layers.dense(self.input, activation=tf.nn.relu, units=32,
                                   kernel_initializer = tf.initializers.glorot_uniform(seed=1),
                                   kernel_regularizer = reg)
            out = tf.layers.dense(lyr1, activation=tf.nn.softmax, units=self.out_shape)
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
        
# Constants     
NUM_CLASSES = 3
NUM_FEATURES = 13
EPOCHS = 10000    
LEARN_RATE = 0.01

## DATA PROCESSING

column_names = ['label', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13']
train, test, train_label, test_label = get_data(column_names)

# one hot encoding
test_label = keras.utils.to_categorical(test_label-1,NUM_CLASSES)
train_label = keras.utils.to_categorical(train_label-1,NUM_CLASSES)

## TRAIN MODEL (Tensorflow)

tf.reset_default_graph()

with tf.Session() as sess:
    mlp = NN('MLP with cross-entropy', sess, NUM_FEATURES, NUM_CLASSES, lr=LEARN_RATE)
    tensor_plot = tf.summary.FileWriter('log/Log_likelihood', graph = sess.graph)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(EPOCHS+1):
        loss, acc = mlp.train(train, train_label)
        
        if epoch % 500 == 0:
            print('TRAIN--Epoch: {}, Loss: {}, Accuracy: {}'.format(epoch, loss, acc))
        
        # get loss plots
        summary = mlp.get_summary(train, train_label)
        # add the loss plot to tensorboard
        tensor_plot.add_summary(summary, epoch)
    
    print('TEST: Accuracy: {}'.format(mlp.test(test, test_label)))

        