import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score


def get_data(column_names):
    data_frame = pd.read_csv('./data/spambase.txt', header=None, sep=',', dtype=np.float)
    data_frame.columns = column_names

    label = data_frame['spam_label'].values
    data = data_frame.drop(labels=['spam_label'], axis=1).values

    return data, label


def normalize(data):
    std = StandardScaler()
    std.fit(data)
    std.transform(data)
    return data


class KNN:

    def __init__(self, kernel):
        self.kernel = kernel
        self.x = None
        self.y = None
        self.pr = None

    def train(self, x_train, y_train):

        # store the train data and labels
        self.x = x_train
        self.y = y_train
        self.pr = self.priors()

    def priors(self):
        classes = np.unique(self.y)

        priors = []

        for label in classes:
            pts = self.y[self.y == label]
            prior = len(pts) / len(self.y)
            priors.append(prior)

        return np.array(priors)

    def get_distance(self, point):
        return np.linalg.norm(self.x - point, axis=1)

    def cosine_kernel(self, point):
        return 1 - self.x.dot(point) / (np.sum(self.x**2, axis=1)**0.5 * (np.sum(point**2)**0.5))

    def gaussian(self, point):
        return np.exp(-1 * np.sum((self.x - point)**2, axis=1))

    def poly_kernel(self, test):
        return (test.dot(self.x.T) ** 2) + 1

    def points(self, test):
        if self.kernel == 'gaussian':
            distances = []

            for entry in test:
                distances.append(self.gaussian(entry))

        elif self.kernel == 'poly':
            distances = self.poly_kernel(test)
        else:
            raise Exception('Please enter valid Kernel choice')

        return np.array(distances)

    def get_probs(self, test):
        distances = self.points(test)
        num_labels = np.unique(self.y)
        label_probs = []

        for i, k in enumerate(distances):
            probs = []

            for label in num_labels:
                p = np.sum(distances[i][self.y == label]) / np.sum(self.y == label)
                probs.append(p)

            label_probs.append(probs)

        return np.array(label_probs)

    def predict(self, x_test):
        predictions = self.get_probs(x_test) * self.pr
        return np.argmax(predictions, axis=1)

    def train_test_accuracy(self, x_test, y_test):

        train_prediction = self.predict(self.x)
        train_acc = accuracy_score(self.y, train_prediction)

        test_prediction = self.predict(x_test)
        test_acc = accuracy_score(y_test, test_prediction)

        return train_acc, test_acc


def train(data, label, kernel, num_splits=10):
    train_accs = []
    test_accs = []

    fold = KFold(n_splits=num_splits, shuffle=True)

    for train_idx, test_idx in fold.split(data, label):
        x_train, y_train = data[train_idx], label[train_idx]
        x_test, y_test = data[test_idx], label[test_idx]

        knn = KNN(kernel)
        knn.train(x_train=x_train, y_train=y_train)

        acc_train, acc_test = knn.train_test_accuracy(x_test, y_test)

        train_accs.append(acc_train)
        test_accs.append(acc_test)

    return train_accs, test_accs


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

np.random.seed(2)
# get data from txt file
data, labels = get_data(column_names)

# normalize data between 0 and 1
data = normalize(data)

with open('./logs/out_prob_spambase', 'w') as file_op:
    train_accs, test_accs = train(data, labels, kernel='gaussian', num_splits=10)

    print('For kernel = {}'.format('gaussian'), file=file_op)

    print('Train accuracies: {}'.format(train_accs), file=file_op)
    print('Mean of train accuracies: {}'.format(np.mean(train_accs)), file=file_op)

    print('Test accuracies: {}'.format(test_accs), file=file_op)
    print('Mean of test accuracies: {}'.format(np.mean(test_accs)), file=file_op)

print('Done!')
