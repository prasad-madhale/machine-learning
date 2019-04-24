
import numpy as np
from sklearn import preprocessing
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy.spatial import distance
from sklearn.metrics import accuracy_score


class ECOC:

    def __init__(self, train_data, train_label, test_data, test_label, coding_matrix):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.c_matrix = coding_matrix

    def train(self):

        train_labels = []

        # create inputs that will go into the multiprocessing code
        for feature_num in range(self.c_matrix.shape[1]):

            # creates a set of what classes should be classified as 1 for this particular adaboost
            feature_set = set()

            # get values from the coding matrix for this feature
            feature_col = self.c_matrix[:, feature_num]

            # add values to the feature set for which value in the coding matrix is 1
            for label_num, val in enumerate(feature_col):
                if val == 1:
                    feature_set.add(label_num)

            # fork copy of train_label as we will be modifying the training labels
            train_label = self.train_label.copy()

            # modify the labels to make it a 0/1 classification problem for adaboost
            train_label = ECOC.modify_labels(train_label, feature_set)

            # create an array of inputs for multiprocessing
            train_labels.append(train_label)

        with Pool(cpu_count()) as pool:
            func = partial(smo, self.train_data)
            models = pool.map(func=func, iterable=train_labels)

            pool.close()
            pool.join()

        # get test accuracy for final model
        test_acc = ECOC.get_accuracy(models, self.test_data, self.test_label, self.c_matrix)

        print('Final Test Accuracy: {}'.format(test_acc))

    @staticmethod
    def get_accuracy(models, data, labels, code_matrix):
        # get prediction on data
        pred = ECOC.predict(models, data, labels)

        # map the code predictions to actual label classes
        pred = ECOC.map_pred_to_classes(pred, code_matrix)

        # get accuracy for predictions given truth
        return accuracy_score(pred, labels)


    @staticmethod
    def predict(models, data, labels):
        final_matrix = []

        for model in models:
            alphas = model[0]
            bias = model[1]

            # get prediction
            prediction = predict(data, labels=labels, alphas=alphas, bias=bias)

            # # convert all -1s to 0s
            # prediction[prediction < 0] = 0

            # append to return
            final_matrix.append(prediction)

        final_matrix = np.array(final_matrix)

        return np.transpose(final_matrix)

    @staticmethod
    def map_pred_to_classes(pred_in_code, code_matrix):
        predict = np.empty(shape=(len(pred_in_code,)))

        # iterate through the final matrix
        for row_num, row in enumerate(pred_in_code):

            min_dist = float('inf')

            # find the code which is closest to the prediction we obtained
            # by using hamming distance as the metric
            for i, c_row in enumerate(code_matrix):
                dist = distance.hamming(row, c_row)

                # store the label class with least hamming distance
                if dist < min_dist:
                    min_dist = dist
                    label = i

            predict[row_num] = label

        return predict

    @staticmethod
    def modify_labels(label, feature_set):

        for ele in range(len(label)):
            if label[ele] in feature_set:
                label[ele] = 1
            else:
                label[ele] = -1

        return label


def normalize(data):
    return preprocessing.minmax_scale(data, feature_range=(0, 1))


def smo(train_data, train_labels, C=1, tol=0.01, max_iter=10, epsilon=0.001):

    train_size = train_data.shape[0]

    # initialize alphas to random values
    # alphas = np.random.random((train_size,))
    alphas = np.zeros(train_size)

    bias = 0
    iter = 0

    func = lambda x: sum([alphas[i] * train_labels[i] * train_data[i, :].dot(x) for i in range(train_size)]) + bias

    # use iterations for stopping criteria
    while iter < max_iter:

        print('Iteration: {}'.format(iter))

        iter += 1

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

    return [alphas, bias]


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

