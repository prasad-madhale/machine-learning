import numpy as np
import pandas as pd
from sklearn import preprocessing


class AdaBoost:

    def __init__(self, col_names):
        self.data, self.labels = AdaBoost.get_data(col_names)
        self.data = AdaBoost.normalize(self.data)

    @staticmethod
    def get_data(names):
        data_frame = pd.read_csv('./data/spambase', sep=',')
        data_frame.columns = names

        labels = data_frame['spam_label'].values

        # replace 0 labels with -1
        labels[labels == 0] = -1

        data_frame = data_frame.drop('spam_label', axis=1)

        return data_frame, labels

    @staticmethod
    def normalize(dataset):
        return preprocessing.minmax_scale(dataset, feature_range=(0, 1))

    def train(self, num_weak_learners=100):
        models = []

        # initialize weights to 1/num_of_data_points
        wts = np.full(self.data.shape[0], (1 / self.data.shape[0]))

        # get all unique thresholds in sorted order
        thresholds = WeakLearner.get_optimal_thresholds(self.data)

        # pick midpoints from the thresholds
        thresholds = WeakLearner.get_midpoints(thresholds)

        for ep in range(num_weak_learners):
            # get predictor using different decision stump selection techniques
            model = WeakLearner.get_tree_predictor(self.data, thresholds, self.labels, wts)
            error = model.error

            # calculate alpha for this model
            model.alpha = 0.5 * np.log((1 - error) / error)

            preds = np.ones(np.shape(self.labels))
            preds[(model.negative * self.data[:, model.feature] <= model.negative * model.threshold)] = -1

            # update wts to assign more importance to incorrect data points
            wts *= np.exp(model.alpha * self.labels * preds)

            wts /= np.sum(wts)

            models.append(model)

            print(model)

class Node:

    def __init__(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold
        self.negative = 1
        self.alpha = None
        self.error = None

    def __str__(self):
        return 'Tree: feature:{} | threshold:{} | negative?:{} | alpha:{}'.format(self.feature, self.threshold,
                                                                                  self.negative, self.alpha)


class WeakLearner:

    @staticmethod
    def get_optimal_thresholds(data):
        thresholds = []

        for i in range(data.shape[1]):
            uniques = np.unique(data[:, i])
            thresholds.append(uniques)

        return np.array(thresholds)

    @staticmethod
    def get_midpoints(data):
        thresholds = []

        for i in range(len(data)):
            thres = data[i]
            thresholds.append((thres[1:] + thres[:-1]) / 2)

        return np.array(thresholds)

    @staticmethod
    def get_tree_predictor(dataset, thresholds, labels, wts):

        min_error = float('inf')

        # check each threshold, feature pair to find the best one
        for feature in range(dataset.shape[1]):
            for threshold in thresholds[feature]:

                # build the weak learner
                tree = Node(feature, threshold)

                # find error for this weak learner
                error = WeakLearner.predict(tree, dataset, labels, wts)

                if error > 0.5:
                    error = 1 - error
                    tree.negative = -1

                # # check our maximization objective
                # diff = np.abs(0.5 - error)

                # find the decision_stump with max diff
                if min_error > error:
                    min_error = error
                    tree.error = min_error
                    best_tree = tree

        return best_tree

    @staticmethod
    def predict(model, test_data, labels, wts):
        feature = model.feature
        threshold = model.threshold

        # initialize with all 1s for predictions
        preds = np.ones(np.shape(labels))

        # change all predictions less than threshold to -1
        preds[test_data[:, feature] <= threshold] = -1

        return np.sum(wts[preds != labels])

    @staticmethod
    def get_random_thresholds(data):
        pass


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

ada_boost = AdaBoost(column_names)
ada_boost.train(10)
