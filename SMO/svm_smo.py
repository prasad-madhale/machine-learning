
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def get_data(names):
    data_frame = pd.read_csv('./data/spambase.txt', header=None, sep=',', dtype=np.float)
    data_frame.columns = names

    label = data_frame['spam_label'].values
    data = data_frame.drop(labels=['spam_label'], axis=1)

    return data, label


def normalize(data):
    std = preprocessing.StandardScaler()
    std.fit(data)
    std.transform(data)

    return data.values


class SVM:

    def __init__(self, c=0.01, tol=0.01, max_pass=5, epsilon=0.001, max_iter=125):
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

        passes = 0
        iteration = 0

        # use iterations for stopping criteria
        while passes < self.max_passes and iteration < self.max_iterations:

            print('Pass: {}'.format(passes))
            print('Iteration: {}'.format(iteration))

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

            # # get accuracy on train data every 10 steps
            # # if iteration % 5 == 0:
            # preds = svm.predict(self.x)
            # t_acc = accuracy_score(self.y, preds)
            # print('Iteration: {} | Train Acc: {}'.format(iteration, t_acc))

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

    def __str__(self):
        return 'SVM with SMO: Max_iterations: {} | Max_passes: {} | C : {} | Tolerance: {} | Epsilon: {}'\
            .format(self.max_iterations, self.max_passes, self.c, self.tolerance, self.eps)


# EXECUTION
with open('./logs/smo_spambase', 'w') as file_op:
    np.random.seed(2)

    # names for the features
    column_names = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our',
                    'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail',
                    'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses',
                    'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you',
                    'word_freq_credit', 'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money',
                    'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab',
                    'word_freq_labs', 'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415',
                    'word_freq_85', 'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm',
                    'word_freq_direct', 'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project',
                    'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;',
                    'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#', 'capital_run_length_average',
                    'capital_run_length_longest', 'capital_run_length_total', 'spam_label']

    # get data from txt file
    df, labels = get_data(column_names)

    # normalize data between 0 and 1
    df = normalize(df)

    # split data into test and train data and labels
    train_data, test_data, train_label, test_label = train_test_split(df, labels, test_size=0.20, random_state=2)

    # modify labels
    train_label = np.where(train_label == 0, -1, train_label)
    test_label = np.where(test_label == 0, -1, test_label)

    # create SMO object
    svm = SVM()

    # train the smo on train data and labels
    svm.fit(train_data, train_label)

    # get train predictions
    train_predictions = svm.predict(train_data)
    train_acc = accuracy_score(train_label, train_predictions)

    # get predictions
    test_predictions = svm.predict(test_data)
    test_acc = accuracy_score(test_label, test_predictions)

    print('Trained SVM using SMO with train-test split (30%)', file=file_op)
    print(svm, file=file_op)

    print('Train Accuracy: {}'.format(train_acc), file=file_op)
    print('Test Accuracy: {}'.format(test_acc), file=file_op)
