
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def get_data(column_names):
    data_frame = pd.read_csv('./data/spambase.txt', header=None, sep=',', dtype=np.float)
    data_frame.columns = column_names

    label = data_frame['spam_label'].values
    data = data_frame.drop(labels=['spam_label'], axis=1)

    return data, label


def normalize(data):
    return preprocessing.minmax_scale(data, feature_range=(0, 1))


def smo(train_data, train_labels, C, tol, max_iter, epsilon=0.001):

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

with open('./logs/smo_spambase', 'w') as file_op:
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
    train_data, test_data, train_label, test_label = train_test_split(df, labels, test_size=0.20, random_state=0)

    # modify labels
    train_label = np.where(train_label == 0, -1, train_label)

    alphas, bias = smo(train_data=train_data, train_labels=train_label, C=1, tol=0.01, max_iter=1000)

    # get predictions
    test_preds = predict(test_data, test_label, alphas, bias)

    print(test_preds)

    # modify back labels
    train_label = np.where(train_label < 0, 0, train_label)

    test_acc = accuracy_score(test_label, test_preds)

    print('Trained SVM using SMO with train-test split (20%)', file=file_op)
    print('Iterations: {} | C: {} | Tolerance: {} | Epsilon: {}'.format(1000, 1, 0.01, 0.001), file=file_op)
    print('Test Accuracy: {}'.format(test_acc), file=file_op)
