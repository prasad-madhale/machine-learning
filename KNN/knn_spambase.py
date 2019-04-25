import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import KFold
from scipy import stats
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

    def __init__(self, k):
        self.k = k
        self.x = None
        self.y = None

    def train(self, x_train, y_train):
        # store the train data and labels
        self.x = x_train
        self.y = y_train

    def get_distance(self, point):
        return np.linalg.norm(self.x - point, axis=1)

    def k_closest_points(self, point):
        distances = self.get_distance(point)
        return np.argsort(distances)[:self.k]

    def predict(self, x_test):
        predictions = []

        for pt in x_test:
            closest_indices = self.k_closest_points(pt)
            closest_pt_labels = self.y[closest_indices]

            pt_prediction = stats.mode(closest_pt_labels)[0][0]
            predictions.append(pt_prediction)

        return predictions

    def train_test_accuracy(self, x_test, y_test):

        train_prediction = self.predict(self.x)
        train_acc = accuracy_score(self.y, train_prediction)

        test_prediction = self.predict(x_test)
        test_acc = accuracy_score(y_test, test_prediction)

        return train_acc, test_acc


def train(K, num_splits=10):
    train_accs = []
    test_accs = []

    fold = KFold(n_splits=num_splits, shuffle=True)

    for train_idx, test_idx in fold.split(df, labels):
        x_train, y_train = df[train_idx], labels[train_idx]
        x_test, y_test = df[test_idx], labels[test_idx]

        knn = KNN(K)
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

# get data from txt file
data, labels = get_data(column_names)

# normalize data between 0 and 1
df = normalize(data)

Ks = [1, 3, 7]

with open('./logs/knn_euclidean', 'w') as file_op:
    for k in Ks:
        train_accs, test_accs = train(k)

        print('For k = {}'.format(k), file=file_op)
        print('Train accuracies: {}'.format(train_accs), file=file_op)
        print('Mean of train accuracies: {}'.format(np.mean(train_accs)), file=file_op)

        print('Test accuracies: {}'.format(test_accs), file=file_op)
        print('Mean of test accuracies: {}'.format(np.mean(test_accs)), file=file_op)
