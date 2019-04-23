
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def get_data(column_names):
    data_frame = pd.read_csv('./data/spambase.txt', header=None, sep=',', dtype=np.float)
    data_frame.columns = column_names

    label = data_frame['spam_label']
    data = data_frame.drop(labels=['spam_label'], axis=1)

    return data, label


def normalize(data):
    return minmax_scale(data, feature_range=(0, 1))


def train_and_test(train_data, test_data, train_label, test_label, clf):
    # train model on the train data and labels
    model = clf.fit(train_data, train_label)

    # predict for train data
    train_predictions = model.predict(train_data)

    # get train accuracy
    train_accuracy = accuracy_score(train_label, train_predictions)

    # predict for test data
    test_predictions = model.predict(test_data)

    # get test accuracy
    test_accuracy = accuracy_score(test_label, test_predictions)

    return train_accuracy, test_accuracy


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
train_data, test_data, train_label, test_label = train_test_split(df, labels, test_size=0.25)

# create SVM classifier with RBF kernel
clf_rbf = svm.SVC(gamma='scale', kernel='rbf', random_state=0)

# create SVM classifier with polynomial kernel
clf_poly = svm.SVC(gamma='scale', kernel='poly', random_state=0)

# train and test the RBF SVM
rbf_train_acc, rbf_test_acc = train_and_test(train_data, test_data, train_label, test_label, clf_rbf)

print('Train Accuracy with RBF Kernel: {}'.format(rbf_train_acc))
print('Test Accuracy with RBF Kernel: {}'.format(rbf_test_acc))

# train and test the Polynomial SVM
poly_train_acc, poly_test_acc = train_and_test(train_data, test_data, train_label, test_label, clf_poly)

print('Train Accuracy with Polynomial Kernel: {}'.format(poly_train_acc))
print('Test Accuracy with Polynomial Kernel: {}'.format(poly_test_acc))


