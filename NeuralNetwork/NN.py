import tensorflow as tf
import pandas as pd
import numpy as np


def get_data(names):
    df = pd.read_csv('./data/wine.txt', sep=',', header=None)
    df.columns = names
    return df


def test_train_split(data, test_percent=50):
    test_size = int(len(data) * test_percent / 100)
    test = data[:test_size]
    train = data[test_size:]

    train_labels = train['label'].values
    test_labels = test['label'].values

    train = train.drop('label', axis=1).values
    test = test.drop('label',axis=1).values

    return train, test, train_labels, test_labels


class NN:
    def __int__(self, name, sess):
        self.scope_name = name
        self.sess = sess

        with(tf.variable_scope('Neural_Network')):
            self.input = tf.placeholder(shape = [None], name='inputs')
            self.labels = tf.placeholder(shape = [None], name='labels')
            self.out = self.build_network(self.input)

        with(tf.variable_scope('loss')):
            loss = tf.losses.mean_squared_error(self.labels, self.out)
            self.train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

    def build_network(self,inputs):
        l1 = tf.layers.dense(inputs, activation = tf.sigmoid, units=32)
        l2 = tf.layers.dense(l1, activation= tf.sigmoid, unit=16)
        out = tf.layers.dense(l2, activation=tf.sigmoid, unit=1)
        return out

    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)

    def train(self, inputs, true_labels, sess):
        return self.sess.run(self.out, feed_dict={self.input:inputs,
                                                  self.labels:true_labels})


column_names = ['label', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13']
dataframe = get_data(column_names)
train_data, test_data, train_label, test_label = test_train_split(dataframe)

# with tf.Session() as sess:
#
#
#     tf.run(tf.global_variables_initializer)
#     nn = NN('NeuralNet', sess)

    # predict = train(train_data, labels,sess)

