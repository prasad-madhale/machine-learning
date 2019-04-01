import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, auc, roc_curve
import random

class AdaBoost:

    def __init__(self, column_names):
        self.data, self.labels = AdaBoost.get_data(column_names)

        # preprocess data
        # replace all occurrences of '?' to the mean if numeric or most frequent
        # value if it's a string
        self.preprocess_data()

    @staticmethod
    def get_data(names):
        data_frame = pd.read_csv('./data/vote/vote.data', sep='\t')

        data_frame.columns = names
        labels = data_frame['label'].values

        # replace labels for adaboost
        # replace d with 1
        labels[labels == 'd'] = 1
        # replace r with -1
        labels[labels == 'r'] = -1

        data_frame = data_frame.drop('label', axis=1)
        return data_frame, np.array(labels, dtype='float64')

    def preprocess_data(self):
        missing_cols = self.data.select_dtypes(include=[object])

        for col in missing_cols.columns:
            indices = self.data[col][self.data[col] == '?'].index.values
            bad_df = self.data.index.isin(indices)

            max_occ = max(Counter(self.data[~bad_df][col]))
            self.data.at[bad_df, col] = max_occ

        self.data = self.data.values

    @staticmethod
    def normalize(dataset):
        return preprocessing.minmax_scale(dataset, feature_range=(0, 1))

    def fit(self, num_weak_learners=100):
        train_data, test_data, train_label, test_label = train_test_split(self.data, self.labels, test_size=0.25,
                                                                          random_state=10)

        # train for the training data
        models, summary = AdaBoost.train(train_data, train_label, test_data, test_label, num_weak_learners)

    @staticmethod
    def train(train_data, train_label, test_data, test_label, num_weak_learners):

        models = []
        summary = np.empty(shape=(num_weak_learners, 4))

        # initialize weights to 1/num_of_data_points (uniform distribution)
        wts = np.full(train_data.shape[0], (1 / train_data.shape[0]))

        # get all unique thresholds in sorted order
        thresholds = WeakLearner.get_thresholds(train_data)

        for ep in range(num_weak_learners):
            # get predictor using different decision stump selection techniques
            model = WeakLearner.get_tree_predictor(train_data, thresholds, train_label, wts)

            # model prediction error
            error = model.error

            # calculate alpha for this model
            model.alpha = 0.5 * np.log((1 - error) / (error + 1e-10))

            preds = np.ones(np.shape(train_label))
            preds[train_data[:, model.feature] <= model.threshold] = -1

            # update wts to assign more importance to incorrect data points
            wts *= np.exp(-model.alpha * train_label * preds)
            wts /= np.sum(wts)

            models.append(model)

            train_error, _ = AdaBoost.get_error(models, train_data, train_label)
            test_error, auc_score = AdaBoost.get_error(models, test_data, test_label)

            # add to summary all the values needed for plotting
            summary[ep] = [model.error, train_error, test_error, auc_score]

            print('Round: {}  |  Feature: {}  |  Threshold: {}  |  Round_error: {}  |  Train_error: {}  |  '
                  'Test_error: {}  |  AUC: {}'.format(ep+1, model.feature, model.threshold, model.error, train_error,
                                                      test_error, auc_score))

        return models, summary

    @staticmethod
    def get_error(models, data, label):
        preds = WeakLearner.get_prediction(models, data)
        acc = accuracy_score(label, preds)

        fpr, tpr, thresholds = roc_curve(label, preds)
        auc_score = auc(fpr, tpr)
        return 1 - acc, auc_score


class Node:

    def __init__(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold
        self.alpha = None
        self.error = None
        self.max_diff = None


class WeakLearner:

    @staticmethod
    def get_thresholds(data):
        thresholds = []

        for i in range(data.shape[1]):
            uniques = np.unique(data[:, i])
            thresholds.append(uniques)

        return np.array(thresholds)

    @staticmethod
    def get_tree_predictor(dataset, thresholds, labels, wts):

        max_diff = -float('inf')

        # pick random feature
        feature = random.randint(0, dataset.shape[1] - 1)

        # pick random threshold
        threshold = np.random.choice(thresholds[feature])

        # build the weak learner
        tree = Node(feature, threshold)

        # find error for this weak learner
        tree.error = WeakLearner.predict(tree, dataset, labels, wts)

        return tree

    @staticmethod
    def predict(model, test_data, labels, wts):
        feature = model.feature
        threshold = model.threshold

        # initialize with all 1s for predictions
        prediction = np.ones(np.shape(labels))

        # change all predictions less than threshold to -1
        prediction[test_data[:, feature] <= threshold] = -1

        return np.sum(wts[prediction != labels])

    @staticmethod
    def get_prediction(models, test_data):

        prediction = np.zeros((test_data.shape[0], ))

        for m in models:
            pred = np.ones(test_data.shape[0])
            pred[test_data[:, m.feature] <= m.threshold] = -1

            prediction += m.alpha * pred

        prediction = np.sign(prediction).flatten()

        return prediction


# EXECUTION
random.seed(2)

column_names = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'label']

ada_boost = AdaBoost(column_names)
ada_boost.fit(num_weak_learners=500)
