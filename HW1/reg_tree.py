import numpy as np
import sys


def get_data():
    train_data = np.genfromtxt('./data/housing_train.txt')
    test_data = np.genfromtxt('./data/housing_test.txt')

    return train_data, test_data


def normalize(data_set, train_len):
    # normalize data using shift/scale
    maxs = np.max(data_set, axis=0)
    mins = np.min(data_set, axis=0)

    maxs = maxs[:-1]
    mins = mins[:-1]

    for feature in range(len(data_set[0])-1):
        for entry in data_set:
            entry[feature] = (entry[feature] - mins[feature]) / (maxs[feature] - mins[feature])

    return data_set[:train_len], data_set[train_len:]


def get_thresholds(dataset):

    ts = []
    for feature in range(len(dataset[0])-1):
        t = []
        for entry in range(len(dataset) - 1):
            t.append((dataset[entry][feature] + dataset[entry+1][feature])/2)
        ts.append(t)

    return np.array(ts)


def get_best_split(dataset, thresholds):
    min_mse = sys.maxsize

    for feature in range(len(dataset[0])-1):
        for threshold in thresholds[feature]:
            left = [entry for entry in dataset if entry[feature] < threshold]
            right = [entry for entry in dataset if entry[feature] >= threshold]

            left_mse = get_mse(left)
            right_mse = get_mse(right)

            mse_after = (left_mse + right_mse) / len(dataset)

            if min_mse > mse_after:
                min_mse = mse_after
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold


def get_mse(dataset):

    if len(dataset) == 0:
        return 0

    prediction = np.mean(dataset, axis=0)
    mse = 0

    for entry in dataset:
        mse += np.square(entry[-1] - prediction[-1])

    return mse / len(dataset)


def build_tree(dataset, prev):

    mse = get_mse(dataset)

    if abs(mse-prev) == 0:
        return

    if len(dataset) < 2:
        return

    thresholds = get_thresholds(dataset)
    feature, threshold = get_best_split(dataset, thresholds)

    print('feature : {}, threshold : {}'.format(feature, threshold))

    left = [entry for entry in dataset if entry[feature] < threshold]
    right = [entry for entry in dataset if entry[feature] >= threshold]

    build_tree(left, mse)
    build_tree(right, mse)


train, test = get_data()
full_data = np.concatenate((train, test), axis=0)
train, test = normalize(full_data, len(train))
thresholds = get_thresholds(train)

build_tree(train, sys.maxsize)

