
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
            svm = SVM()
            func = partial(svm.fit, self.train_data)
            models = pool.map(func=func, iterable=train_labels)

            pool.close()
            pool.join()

        # get test accuracy for final model
        test_acc = self.get_accuracy(models, self.test_data, self.test_label, self.c_matrix)

        print('Final Test Accuracy: {}'.format(test_acc))

    def get_accuracy(self, models, x_test, y_test, code_matrix):

        # get prediction on data
        pred = ECOC.predict(models, self.train_data, self.train_label, x_test)

        # map the code predictions to actual label classes
        pred = ECOC.map_pred_to_classes(pred, code_matrix)

        # get accuracy for predictions given truth
        return accuracy_score(y_test, pred)

    @staticmethod
    def predict(models, data, labels, x_test):
        final_matrix = []

        for model in models:
            alphas = model[0]
            bias = model[1]

            # get prediction
            prediction = SVM.predict_static(alphas, bias, data, labels, x_test)

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


class SVM:

    def __init__(self, c=0.01, tol=0.01, max_pass=1, epsilon=0.001, max_iter=1):
        self.c = c
        self.tolerance = tol
        self.max_passes = max_pass
        self.eps = epsilon
        self.max_iterations = max_iter
        self.x = None
        self.y = None
        self.alphas = None
        self.bias = 0

    def fit(self, data, label):

        self.x = data
        self.y = label

        train_size = self.x.shape[0]

        # initialize alphas to random values
        # alphas = np.random.random((train_size,))
        self.alphas = np.zeros(train_size)
        self.bias = 0

        passes = 0
        iteration = 0

        # use iterations for stopping criteria
        while passes < self.max_passes and iteration < self.max_iterations:

            iteration += 1

            # count of updated alphas
            num_changed_alphas = 0

            for i in range(train_size):
                ei = self.func(self.x[i]) - self.y[i]

                num = (self.y[i] * ei)

                if (num < -self.tolerance and self.alphas[i] < self.c) or (num > self.tolerance and self.alphas[i] > 0):

                    # choose j such that j is not equal to i
                    j = np.random.choice(np.concatenate([np.arange(i), np.arange(i+1, train_size)]), size=1)[0]

                    ej = self.func(self.x[j]) - self.y[j]

                    # save old alphas by making a copy of them
                    alpha_old = self.alphas.copy()

                    # compute lower and upper bounds
                    low, high = self.compute_bounds(i, j)

                    if low == high:
                        continue

                    # compute eta
                    eta = 2 * SVM.lin_kernel(self.x[i], self.x[j]) - SVM.lin_kernel(self.x[i], self.x[i]) - \
                          SVM.lin_kernel(self.x[j], self.x[j])

                    if eta == 0:
                        continue

                    # update alphas
                    self.alphas[j] = self.alphas[j] - ((self.y[j] * (ei - ej)) / eta)

                    # clip alphas between the bounds
                    self.alphas[j] = SVM.bound_alpha(self.alphas[j], low, high)

                    if np.abs(self.alphas[j] - alpha_old[j]) < self.eps:
                        continue

                    self.alphas[i] = self.alphas[i] + self.y[i] * self.y[j] * (alpha_old[j] - self.alphas[j])

                    b1 = self.bias - ei - (self.y[i] * (self.alphas[i] - alpha_old[i]) *
                                           SVM.lin_kernel(self.x[i], self.x[i])) - \
                         (self.y[j] * (self.alphas[j] - alpha_old[j]) * SVM.lin_kernel(self.x[i], self.x[j]))

                    b2 = self.bias - ej - (self.y[i] * (self.alphas[i] - alpha_old[i]) * SVM.lin_kernel(self.x[i],
                                                                                                         self.x[j])) - \
                         (self.y[j] * (self.alphas[j] - alpha_old[j]) * SVM.lin_kernel(self.x[j], self.x[j]))

                    # update bias using b1 and b2
                    self.bias = self.select_bias(b1, b2, i, j)

                    # increment count of changed alphas
                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

        # get accuracy on train data
        preds = self.predict(self.x)
        t_acc = accuracy_score(self.y, preds)
        print('Individual SVM Train Acc: {}'.format(t_acc))

        return [self.alphas, self.bias]

    def select_bias(self, b1, b2, i, j):

        if 0 < self.alphas[i] < self.c:
            bias = b1
        elif 0 < self.alphas[j] < self.c:
            bias = b2
        else:
            bias = (b1 + b2) / 2

        return bias

    @staticmethod
    def bound_alpha(alpha_j, lower, higher):
        if alpha_j > higher:
            alpha_j = higher
        elif alpha_j < lower:
            alpha_j = lower

        return alpha_j

    def func(self, xi):
        return (self.alphas * self.y * SVM.lin_kernel(xi, self.x)).sum() + self.bias

    @staticmethod
    def lin_kernel(xi, x):
        return np.dot(x, xi)

    def compute_bounds(self, i, j):
        if self.y[i] != self.y[j]:
            lower = max(0, self.alphas[j] - self.alphas[i])
            higher = min(self.c, self.c + self.alphas[j] - self.alphas[i])
        else:
            lower = max(0, self.alphas[i] + self.alphas[j] - self.c)
            higher = min(self.c, self.alphas[i] + self.alphas[j])

        return lower, higher

    def predict(self, x_test):
        predictions = []

        for entry in range(x_test.shape[0]):
            val = self.func(x_test[entry])

            if val < 0:
                predictions.append(-1)
            else:
                predictions.append(1)

        return np.array(predictions)

    @staticmethod
    def predict_static(alphas, bias, x_train, y_train, x_test):
        predictions = []

        for entry in range(x_test.shape[0]):
            val = SVM.func_static(alphas, bias, x_train, y_train, x_test[entry])

            if val < 0:
                predictions.append(-1)
            else:
                predictions.append(1)

        return np.array(predictions)

    @staticmethod
    def func_static(alphas, bias, x, y, xi):
        return (alphas * y * SVM.lin_kernel(xi, x)).sum() + bias

    def __str__(self):
        return 'SVM with SMO: Max_iterations: {} | Max_passes: {} | C : {} | Tolerance: {} | Epsilon: {}'\
            .format(self.max_iterations, self.max_passes, self.c, self.tolerance, self.eps)