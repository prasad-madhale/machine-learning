
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from Ecoc import ECOC


def get_data(column_names):
    data_frame = pd.read_csv('./data/spambase.txt', header=None, sep=',', dtype=np.float)
    data_frame.columns = column_names

    label = data_frame['spam_label'].values
    data = data_frame.drop(labels=['spam_label'], axis=1)

    return data, label


def normalize(data):
    return preprocessing.minmax_scale(data, feature_range=(0, 1))


# EXECUTION
mnist = tf.keras.datasets.mnist

# get training and testing mnist set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_haar = np.load(file='./data/train_haar_features.npz')['train_haar']
test_haar = np.load(file='./data/test_haar_features.npz')['test_haar']
train_labels = np.load(file='./data/train_labels.npz')['train_labels']
test_labels = y_test

with open('./logs/out_smo_digits', 'w') as file_op:
    print('HAAR features loaded!', file=file_op)
    print('SHAPES:', file=file_op)
    print(train_haar.shape, test_haar.shape, train_labels.shape, test_labels.shape, file=file_op)

    # generate coding matrix for ECOC procedure
    coding_matrix = np.array([[1,0,1,0,0,1,1,0,0,1,1,0,1,0,1,1,1,1,1,0],
                            [1,0,0,0,0,1,0,1,0,0,1,1,0,0,1,1,0,1,0,1],
                            [1,1,0,0,0,0,1,0,1,0,0,1,1,0,0,1,1,0,1,0],
                            [1,1,1,0,0,0,0,1,0,1,0,0,1,1,0,0,1,1,0,1],
                            [1,1,1,1,0,0,0,0,1,0,1,0,0,1,1,0,0,1,1,0],
                            [1,1,1,1,1,0,0,0,0,1,0,1,0,0,1,1,0,0,1,1],
                            [0,1,1,1,1,1,0,0,0,0,1,0,1,0,0,1,1,0,0,1],
                            [1,0,1,1,1,1,1,0,0,0,0,1,0,1,0,0,1,1,0,0],
                            [0,1,0,1,0,0,1,1,0,0,1,1,0,1,0,1,1,1,1,1],
                            [0,0,1,0,1,0,0,1,1,0,0,1,1,0,1,0,1,1,1,1]])

    ecoc = ECOC(train_haar, train_labels, test_haar, test_labels, coding_matrix)
    ecoc.train()



