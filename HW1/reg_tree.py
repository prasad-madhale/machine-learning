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
            left, right = split_data(dataset, feature, threshold)

            if len(left) == 0 or len(right) == 0:
                continue

            left_mse = get_mse(left)
            right_mse = get_mse(right)

            mse_after = (left_mse + right_mse) / len(dataset)

            if min_mse > mse_after:
                min_mse = mse_after
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold


class Terminal:

    def __init__(self, dataset):
        predicts = get_prediction(dataset)
        self.prediction = predicts[-1]

    def predict(self):
        return self.prediction


class Node:

    def __init__(self, feature, threshold, left_node, right_node):
        self.feature = feature
        self.threshold = threshold
        self.left_node = left_node
        self.right_node = right_node


def get_prediction(dataset):
    return np.mean(dataset, axis=0)


def split_data(dataset, feature, threshold):
    left = [entry for entry in dataset if entry[feature] < threshold]
    right = [entry for entry in dataset if entry[feature] >= threshold]
    return left, right


def get_mse(dataset):
    if len(dataset) == 0:
        return 0

    prediction = get_prediction(dataset)
    mse = 0

    for entry in dataset:
        mse += np.square(entry[-1] - prediction[-1])

    return mse / len(dataset)


def build_tree(dataset):

    if get_mse(dataset) <= 1e-4 or len(dataset) <= 2:
        return Terminal(dataset)

    thresholds = get_thresholds(dataset)
    best_feature, best_threshold = get_best_split(dataset, thresholds)

    left_data, right_data = split_data(dataset, best_feature, best_threshold)

    left_node = build_tree(left_data)
    right_node = build_tree(right_data)

    return Node(best_feature, best_threshold, left_node, right_node)


def print_tree(root):

    if isinstance(root, Terminal):
        print('Prediction: {}'.format(root.prediction))
        return

    print('({}, {})'.format(root.feature, root.threshold))

    print('Left')
    print_tree(root.left_node)
    print('Right')
    print_tree(root.right_node)

    return


train, test = get_data()
full_data = np.concatenate((train, test), axis=0)
train, test = normalize(full_data, len(train))
thresholds = get_thresholds(train)

model = build_tree(train)
print_tree(model)

