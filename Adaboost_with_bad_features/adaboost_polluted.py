import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import random


class AdaBoost:

    def __init__(self):
        self.train_data, self.train_labels, self.test_data, self.test_labels = AdaBoost.get_data()
        self.train_data, self.test_data = AdaBoost.normalize(self.train_data, self.test_data)

    @staticmethod
    def get_data():
        test_data = pd.read_csv('./data/spam_polluted/test_feature.txt', sep=' ')
        test_label = pd.read_csv('./data/spam_polluted/test_label.txt', sep='\n')
        train_data = pd.read_csv('./data/spam_polluted/train_feature.txt', sep=' ')
        train_label = pd.read_csv('./data/spam_polluted/train_label.txt', sep='\n')

        # flatten labels
        test_label = test_label.values.flatten()
        train_label = train_label.values.flatten()

        # change all 0s to -1s
        test_label[test_label == 0] = -1
        train_label[train_label == 0] = -1

        return train_data, train_label, test_data, test_label

    @staticmethod
    def normalize(train_data, test_data):
        # combine data
        full_data = np.concatenate([train_data, test_data])

        test_data_size = len(train_data)

        # normalize data
        full_data = preprocessing.minmax_scale(full_data, feature_range=(0, 1))

        train_data = full_data[:test_data_size]
        test_data = full_data[test_data_size:]

        return train_data, test_data

    def fit(self, num_weak_learners=100):
        # train for the training data
        models, summary = AdaBoost.train(self.train_data, self.train_labels, self.test_data, self.test_labels,
                                         num_weak_learners)

        # plot round errors
        AdaBoost.plot_round_error(summary[:, 0])

        # plot train/test errors
        AdaBoost.plot_train_test_error(summary[:, 1], summary[:, 2])

        # plot test AUC scores
        AdaBoost.plot_auc(summary[:, 3])

        # get predictions for test data using final models
        test_predictions = WeakLearner.get_prediction(models, self.test_data)

        # plot roc
        AdaBoost.plot_roc(self.test_labels, test_predictions)

    @staticmethod
    def train(train_data, train_label, test_data, test_label, num_weak_learners):

        models = []
        summary = np.empty(shape=(num_weak_learners, 4))

        # initialize weights to 1/num_of_data_points (uniform distribution)
        wts = np.full(train_data.shape[0], (1 / train_data.shape[0]))

        # get all unique thresholds in sorted order
        thresholds = WeakLearner.get_thresholds(train_data)

        # pick midpoints from the thresholds
        thresholds = WeakLearner.get_midpoints(thresholds)

        for ep in range(num_weak_learners):
            # get predictor using different decision stump selection techniques
            model = WeakLearner.get_tree_predictor(train_data, thresholds, train_label, wts)

            # model prediction error
            error = model.error

            # calculate alpha for this model
            model.alpha = 0.5 * np.log((1 - error) / (error + 1e-15))

            preds = np.ones(np.shape(train_label))
            preds[train_data[:, model.feature] <= model.threshold] = -1

            # update wts to assign more importance to incorrect data points
            wts *= np.exp(-model.alpha * train_label * preds)
            wts /= np.sum(wts)

            models.append(model)

            train_error, _ = AdaBoost.get_error(models, train_data, train_label)
            test_error, test_acc = AdaBoost.get_error(models, test_data, test_label)

            # add to summary all the values needed for plotting
            summary[ep] = [model.error, train_error, test_error, test_acc]

            print('Round: {}  |  Feature: {}  |  Threshold: {}  |  Round_error: {}  |  Train_error: {}  |   '
                  'Test_error: {}  |  Test Accuracy: {}'.format(ep, model.feature, model.threshold, model.error,
                                                                train_error, test_error, test_acc), file=f)

        return models, summary

    @staticmethod
    def get_error(models, data, label):
        preds = WeakLearner.get_prediction(models, data)
        acc = accuracy_score(label, preds)
        return 1 - acc, acc

    @staticmethod
    def plot_round_error(round_errors):
        plt.xlabel('Iteration Step')
        plt.ylabel('Round Error')
        plt.title('Round Error')
        plt.plot(round_errors)
        plt.savefig('./plots/round_error.png')
        plt.close()

    @staticmethod
    def plot_train_test_error(train_error, test_error):
        num_itrs = train_error.shape[0]
        itrs = np.arange(num_itrs)

        plt.xlabel('Iteration Step')
        plt.ylabel('Train/Test Error (Red/Blue)')
        plt.title('Train/Test Error (Red/Blue)')

        plt.plot(itrs, train_error, 'r-', label='Train Error')
        plt.plot(itrs, test_error, 'b-', label='Test Error')

        # add legend
        plt.legend(loc='upper right')

        plt.savefig('./plots/train_test_error.png')
        plt.close()

    @staticmethod
    def plot_auc(aucs):
        plt.xlabel('Iteration Step')
        plt.ylabel('AUC')
        plt.title('AUC')
        plt.plot(aucs)
        plt.savefig('./plots/auc.png')
        plt.close()

    @staticmethod
    def plot_roc(truth, preds):
        fprs, tprs, _ = roc_curve(truth, preds)

        plt.title('ROC')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        plt.plot(fprs, tprs, label='AUC: {}'.format(auc(fprs, tprs)))
        plt.legend(loc='lower right')

        plt.savefig('./plots/roc.png')
        plt.close()


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
            uniqs = np.unique(data[:, i])
            uniqs = np.sort(uniqs)
            thresholds.append(uniqs)

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
        max_diff = -float('inf')

        # check each threshold, feature pair to find the best one
        for feature in range(dataset.shape[1]):
            for threshold in thresholds[feature]:

                # build the weak learner
                tree = Node(feature, threshold)

                # find error for this weak learner
                error = WeakLearner.predict(tree, dataset, labels, wts)

                diff = np.abs(0.5 - error)

                if diff > max_diff:
                    max_diff = diff
                    tree.error = error
                    tree.max_diff = max_diff
                    best_tree = tree

        return best_tree

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


# set seed for random operations
np.random.seed(2)
random.seed(2)

with open('./logs/out_polluted.txt', 'w') as f:
    ada_boost = AdaBoost()
    ada_boost.fit(num_weak_learners=15)

print('Done Training!')
