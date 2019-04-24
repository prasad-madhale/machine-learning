
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


def smo(train_data, train_labels, C, tol, max_iter, epsilon=10e-5):

    train_size = train_data.shape[0]

    # initialize alphas to random values
    # alphas = np.random.random((train_size,))
    alphas = np.zeros(train_size)

    bias = 0
    iter = 0

    func = lambda x: sum([alphas[i] * train_labels[i] * train_data[i, :].dot(x) for i in range(train_size)]) + bias

    # use iterations for stopping criteria
    while iter < max_iter:

        # count of updated alphas
        num_changed_alphas = 0

        for i in range(train_size):
            ei = func(train_data[i, :]) - train_labels[i]

            if ((train_labels[i] * ei) < -tol and alphas[i] < C) or ((train_labels[i] * ei) > tol and alphas[i] > 0):

                # choose j such that j is not equal to i
                j = np.random.choice(np.concatenate([np.arange(i), np.arange(i+1, train_size)]), size=1)[0]

                ej = func(train_data[i, :]) - train_labels[j]

                # save old alphas by making a copy of them
                alpha_old = alphas.copy()

                # compute lower and upper bounds
                L, H = compute_bounds(train_labels, i, j, alphas, C)

                if L == H:
                    continue

                # compute eta
                eta = 2 * train_data[i].dot(train_data[j]) - train_data[i].dot(train_data[i]) - train_data[j].dot(train_data[j])

                if eta == 0:
                    continue

                # update alphas
                alphas = alphas - ((train_labels[j] * (ei - ej)) / eta)

                # clip alphas between the bounds
                alphas[j] = bound_alpha(alphas[j], L, H)

                if np.abs(alphas[j] - alpha_old[j]) < epsilon:
                    continue

                alphas[i] = alphas[i] + train_labels[i] * train_labels[j] * (alpha_old[j] - alphas[j])

                b1 = (bias - ei - (train_labels[i] * (alphas[i] - alpha_old[i]) * (train_data[i].dot(train_data[i])))) - \
                     (train_labels[j] * (alphas[j] - alpha_old[j]) * (train_data[i].dot(train_data[j])))

                b2 = (bias - ej - (train_labels[i] * (alphas[i] - alpha_old[i]) * (train_data[i].dot(train_data[j])))) - \
                     (train_labels[j] * (alphas[j] - alpha_old[j]) * (train_data[j].dot(train_data[j])))

                # update bias using b1 and b2
                bias = select_bias(b1, b2, alphas, i, j, C)

                # increment count of changed alphas
                num_changed_alphas += 1

        if num_changed_alphas == 0:
            iter += 1
        else:
            iter = 0

        print('Iteration: {}'.format(iter))

    return alphas, bias


def select_bias(b1, b2, alphas, i, j, C):

    if 0 < alphas[i] < C:
        bias = b1
    elif 0 < alphas[j] < C:
        bias = b2
    else:
        bias = (b1 + b2) / 2

    return bias


def bound_alpha(alphaJ, L, H):
    if alphaJ > H:
        alphaJ = min(H, alphaJ)
    elif alphaJ < L:
        alphaJ = max(L, alphaJ)

    return alphaJ


def compute_bounds(train_labels, i, j, alphas, C):
    if train_labels[i] != train_labels[j]:
        lower = max(0, alphas[j] - alphas[i])
        higher = min(C, C + alphas[j] - alphas[i])
    else:
        lower = max(0, alphas[i] + alphas[j] - C)
        higher = min(C, alphas[i] + alphas[j])

    return lower, higher


def predict(data, labels, alphas, bias):
    func = lambda x: sum([alphas[i] * labels[i] * data[i, :].dot(x) for i in range(len(labels))]) + bias

    predictions = []

    for entry in range(data.shape[0]):
        predictions.append((func(data[entry, :]) > 0))

    return np.array(predictions)


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



