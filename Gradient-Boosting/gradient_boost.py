import numpy as np
import pandas as pd


class RegressionTree:

    def __init__(self, col_names):
        self.train_data, self.test_data = RegressionTree.get_data(col_names)
        full_data = pd.concat([self.train_data, self.test_data])
        self.train_data, self.test_data = RegressionTree.normalize(full_data, len(self.train_data))
        self.model = None

    # retrieve data from the given location
    @staticmethod
    def get_data(names):
        train_dataframe = pd.read_csv('./data/housing_train.txt', delim_whitespace=True, header=None)
        test_dataframe = pd.read_csv('./data/housing_test.txt', delim_whitespace=True, header=None)
        train_dataframe.columns = names
        test_dataframe.columns = names

        return train_dataframe, test_dataframe

    @staticmethod
    def normalize(dataset, train_len):

        # normalize data using shift/scale
        maxs = dataset.max()
        mins = dataset.min()

        for feature in dataset.columns[:-1]:
            for i, entry in dataset.iterrows():
                dataset.at[i, feature] = (entry[feature] - mins[feature]) / (maxs[feature] - mins[feature])

        return dataset.iloc[:train_len], dataset.iloc[train_len:]

    # @staticmethod
    # def normalize(dataset, train_len):
    #     cols = dataset.columns[:-1]
    #     dataset[cols] = dataset[cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    #
    #     return dataset.iloc[:train_len], dataset.iloc[train_len:]

    @staticmethod
    def get_thresholds(dataset, feature):
        dataset.sort_values(by=[feature])
        ts = []

        for entry in range(len(dataset) - 1):
            ts.append((dataset.iloc[entry][feature] + dataset.iloc[entry + 1][feature]) / 2)

        return ts

    # @staticmethod
    # def get_thresholds(dataset, feature):
    #     return np.unique(dataset.iloc[:][feature])

    @staticmethod
    def get_best_split(dataset):

        max_info_gain = 0
        best_feature = None
        best_threshold = None
        mse_before = RegressionTree.get_mse(dataset)

        for feature in dataset.columns[:-1]:

            thresholds = RegressionTree.get_thresholds(dataset, feature)

            for threshold in thresholds:
                left, right = RegressionTree.split_data(dataset, feature, threshold)

                if len(left) == 0 or len(right) == 0:
                    continue

                left_mse = RegressionTree.get_mse(left)
                right_mse = RegressionTree.get_mse(right)

                w = len(left) / len(dataset)
                mse_after = (w * left_mse) + ((1 - w) * right_mse)

                info_gain = mse_before - mse_after

                if max_info_gain <= info_gain:
                    max_info_gain = info_gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, max_info_gain

    @staticmethod
    def get_mse(dataset):

        prediction = dataset['MEDV'].mean()
        errors = []

        for i, entry in dataset.iterrows():
            errors.append(np.square(entry['MEDV'] - prediction))

        mse = pd.Series(errors).mean()
        return mse

    @staticmethod
    def split_data(dataset, feature, threshold):

        left = dataset[dataset[feature] < threshold]
        right = dataset[dataset[feature] >= threshold]

        return left, right

    def fit(self, max_depth=2):
        self.model = RegressionTree.build_tree(self.train_data, 0, max_depth)

    @staticmethod
    def build_tree(dataset, depth, max_depth):

        best_feature, best_threshold, info_gain = RegressionTree.get_best_split(dataset)

        if info_gain == 0 or depth >= max_depth:
            return Terminal(dataset)

        print('Split Selected: (Feature: {}, Threshold: {}, Info Gain: {})'.format(best_feature, best_threshold,
                                                                                   info_gain))

        left_data, right_data = RegressionTree.split_data(dataset, best_feature, best_threshold)

        left_node = RegressionTree.build_tree(left_data, depth + 1, max_depth)
        right_node = RegressionTree.build_tree(right_data, depth + 1, max_depth)

        return Node(best_feature, best_threshold, left_node, right_node)

    @staticmethod
    def regress(root, entry):

        if isinstance(root, Terminal):
            return root.predict()

        if entry[root.feature] < root.threshold:
            result = RegressionTree.regress(root.left_node, entry)
        else:
            result = RegressionTree.regress(root.right_node, entry)

        return result

    def test(self):
        train_error, _ = RegressionTree.test_model(self.model, self.train_data)
        test_error, _ = RegressionTree.test_model(self.model, self.test_data)
        return train_error, test_error

    def predict(self, identifier):
        if identifier == 'train':
            mse, prediction = RegressionTree.test_model(self.model, self.train_data)
        elif identifier == 'test':
            mse, prediction = RegressionTree.test_model(self.model, self.test_data)

        return mse, prediction

    @staticmethod
    def test_model(model, test_data):

        predictions = []

        for i, entry in test_data.iterrows():
            predictions.append(RegressionTree.regress(model, entry))

        errors = []

        for i, p in enumerate(predictions):
            errors.append(np.square(test_data.iloc[i]['MEDV'] - p))

        mse = pd.Series(errors).mean()
        return mse, np.array(predictions)

    def update_labels(self, identifier, preds):
        if identifier == 'train':
            self.train_data.iloc[:][self.train_data.columns[-1]] -= preds
        elif identifier == 'test':
            self.test_data.iloc[:][self.test_data.columns[-1]] -= preds


class Terminal:

    def __init__(self, dataset):
        self.prediction = dataset['MEDV'].mean()

    def predict(self):
        return self.prediction


class Node:

    def __init__(self, feature, threshold, left_node, right_node):
        self.feature = feature
        self.threshold = threshold
        self.left_node = left_node
        self.right_node = right_node


class GradientBoosting:

    def __init__(self, iters, column_names):
        self.max_iters = iters
        self.col_names = column_names

    def fit(self, max_depth):
        reg_tree = RegressionTree(self.col_names)

        for itr in range(self.max_iters):
            # train a decision tree
            reg_tree.fit(max_depth)

            _, train_predictions = reg_tree.predict('train')
            _, test_predictions = reg_tree.predict('test')

            train_error, test_error = reg_tree.test()

            # print training mse at each iteration
            print('Training Error at {}: {}'.format(itr+1, train_error))
            print('Testing Error at {}: {}'.format(itr+1, test_error))

            # update the labels using residues
            # for train
            reg_tree.update_labels('train', train_predictions)

            # for test
            reg_tree.update_labels('test', test_predictions)


# EXECUTION
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
g_boost = GradientBoosting(5, column_names)
g_boost.fit(2)

