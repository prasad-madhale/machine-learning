import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.metrics import balanced_accuracy_score


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

    def __init__(self, radius, kernel):
        self.radius = radius
        self.kernel = kernel
        self.x = None
        self.y = None

    def train(self, x_train, y_train):

        # store the train data and labels
        self.x = x_train
        self.y = y_train

    def get_distance(self, point):
        return np.linalg.norm(self.x - point, axis=1)

    def cosine_kernel(self, point):
        return 1 - self.x.dot(point) / (np.sum(self.x**2, axis=1)**0.5 * (np.sum(point**2)**0.5))

    def k_closest_points(self, point):
        if self.kernel == 'euclidean':
            distances = self.get_distance(point)
        elif self.kernel == 'cosine':
            distances = self.cosine_kernel(point)
        else:
            raise Exception('Please enter valid Kernel choice')

        return distances < self.radius

    def predict(self, x_test):
        predictions = []

        for pt in x_test:
            closest_indices = self.k_closest_points(pt)
            closest_pt_labels = self.y[closest_indices]

            # if no points within the radius we return majority label
            try:
                pt_prediction = stats.mode(closest_pt_labels)[0][0]
            except:
                # pt_prediction = stats.mode(self.y)[0][0]
                pt_prediction = self.y[np.argmin(closest_indices)]
            predictions.append(pt_prediction)

        return predictions

    def train_test_accuracy(self, x_test, y_test):

        train_prediction = self.predict(self.x)
        train_acc = balanced_accuracy_score(self.y, train_prediction)

        test_prediction = self.predict(x_test)
        test_acc = balanced_accuracy_score(y_test, test_prediction)

        return train_acc, test_acc


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

# 5.306
# 5.353

np.random.seed(2)
RADIUS = 4.408163265306122

# get data from txt file
data, labels = get_data(column_names)

# normalize data between 0 and 1
data = normalize(data)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=2)

with open('./logs/out_knn_radius_spambase', 'w') as file_op:
    knn = KNN(RADIUS, kernel='euclidean')
    knn.train(x_train, y_train)

    train_accs, test_accs = knn.train_test_accuracy(x_test, y_test)

    print('For kernel = {}'.format(knn.kernel), file=file_op)
    print('For radius = {}'.format(knn.radius), file=file_op)
    print('Train Accuracy: {}'.format(train_accs), file=file_op)
    print('Test Accuracy: {}'.format(test_accs), file=file_op)

