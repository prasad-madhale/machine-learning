import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


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


def get_data(names):
    data_frame = pd.read_csv('./data/spambase.txt', header=None, sep=',', dtype=np.float)
    data_frame.columns = names

    label = data_frame['spam_label'].values
    data = data_frame.drop(labels=['spam_label'], axis=1).values

    return data, label


def normalize(data):
    std = StandardScaler()
    std.fit(data)
    std.transform(data)
    return data


def get_distance(x, point):
    return np.linalg.norm(x - point, axis=1)


def select_features(x, y, num_features):
    wts = np.zeros(x.shape[1])

    for idx, point in enumerate(x):
        # get distance of point with each point in x
        distance = get_distance(x, point)

        # sort distances and get least indices
        indices = np.argsort(distance)

        same_z = x[np.argsort(indices[y == y[idx]])[1]]
        opp_z = x[np.argmin(indices[y != y[idx]])]

        # update weights
        wts -= (point - same_z)**2 + (point - opp_z)**2

    sorted_features = np.argsort(wts)[::-1]

    return sorted_features[:num_features]


# EXECUTION
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

column_names = np.array(column_names)

with open('./logs/knn_relief', 'w') as file_op:
    # get data from txt file
    data, labels = get_data(column_names)

    # normalize data between 0 and 1
    data = normalize(data)

    # split data into test and train
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=2)

    print('Original Data shapes:', file=file_op)
    print('x-train: {} x-test: {} y-train: {}, y-test: {}'
          .format(x_train.shape, x_test.shape, y_train.shape, y_test.shape), file=file_op)

    # get the top features
    selected_features = select_features(x_train, y_train, 5)

    print('Top 5 features: {}'.format(column_names[selected_features]), file=file_op)

    # use only top selected features
    x_train = x_train[:, selected_features]
    x_test = x_test[:, selected_features]

    print('New Data shapes:', file=file_op)
    print('x-train: {} x-test: {} y-train: {}, y-test: {}'
          .format(x_train.shape, x_test.shape, y_train.shape, y_test.shape), file=file_op)

    knn = KNN(k=3)
    knn.train(x_train, y_train)

    train_acc, test_acc = knn.train_test_accuracy(x_test, y_test)

    print('Train Accuracy: {}'.format(train_acc), file=file_op)
    print('Test Accuracy: {}'.format(test_acc), file=file_op)