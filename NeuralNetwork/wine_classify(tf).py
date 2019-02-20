#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:44:04 2019

@author: prasad
"""

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

def get_data(names):
    df = pd.read_csv('./data/wine.txt', sep=',', header=None)
    df.columns = names
    return df


def split(df):
    labels = df['label'].values
    data = df.drop('label', axis=1).values

    return data, labels

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
            self.loss = tf.losses.mean_squared_error(self.labels, self.predict)
            self.loss_plot = tf.summary.scalar('loss', self.loss)
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        
        self.summaries = tf.summary.merge_all()
        
        with tf.variable_scope('accuracy'):
            _, self.acc = tf.metrics.accuracy(tf.math.argmax(self.labels,1),
                                              tf.math.argmax(self.predict,1))
        
    def build_network(self):
        with tf.variable_scope('mlp'):
            reg = keras.regularizers.l2(0.1)
            lyr1 = tf.layers.dense(self.input, activation=tf.nn.sigmoid, units=32,
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

## DATA PROCESSING

column_names = ['label', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13']
dataframe = get_data(column_names)

# shuffle data
dataframe = dataframe.sample(frac = 1)

data, label = split(dataframe)

# normalize
data = keras.utils.normalize(data, axis=1, order=2)

# split data into test and train
train_data, test_data, train_label, test_label = train_test_split(data, label, test_size = 0.5)

# one hot encoding
test_label = keras.utils.to_categorical(test_label-1,NUM_CLASSES)
train_label = keras.utils.to_categorical(train_label-1,NUM_CLASSES)


## TRAIN MODEL (Tensorflow)

tf.reset_default_graph()

with tf.Session() as sess:
    mlp = NN('MLP', sess, NUM_FEATURES, NUM_CLASSES, lr=0.01)
    tensor_plot = tf.summary.FileWriter('log/Classification', graph = sess.graph)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(EPOCHS+1):
        loss, acc = mlp.train(train_data, train_label)
        
        if epoch % 500 == 0:
            print('TRAIN--Epoch: {}, Loss: {}, Accuracy: {}'.format(epoch, loss, acc))
        
        # get loss plots
        summary = mlp.get_summary(train_data, train_label)
        # add the loss plot to tensorboard
        tensor_plot.add_summary(summary, epoch)
    
    print('TEST: Accuracy: {}'.format(mlp.test(test_data, test_label)))

        