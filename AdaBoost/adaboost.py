import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, auc, roc_curve
import matplotlib.pyplot as plt


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

    def fit(self, num_weak_learners=100):
        train_data, test_data, train_label, test_label = train_test_split(self.data, self.labels, test_size=0.25)

        # train for the training data
        models, summary = AdaBoost.train(train_data, train_label, test_data, test_label, num_weak_learners)

        # plot round errors
        AdaBoost.plot_round_error(summary[:, 0])

        # plot train/test errors
        AdaBoost.plot_train_test_error(summary[:, 1], summary[:, 2])

        # plot test AUC scores
        AdaBoost.plot_auc(summary[:, 3])

        # get predictions for test data using final models
        test_predictions = WeakLearner.get_prediction(models, test_data)

        # plot roc
        AdaBoost.plot_roc(test_label, test_predictions)

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

    @staticmethod
    def train(train_data, train_label, test_data, test_label, num_weak_learners):

        models = []
        summary = np.empty(shape=(num_weak_learners, 4))

        # initialize weights to 1/num_of_data_points (uniform distribution)
        wts = np.full(train_data.shape[0], (1 / train_data.shape[0]))

        # get all unique thresholds in sorted order
        thresholds = WeakLearner.get_optimal_thresholds(train_data)

        # pick midpoints from the thresholds
        thresholds = WeakLearner.get_midpoints(thresholds)

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
ada_boost.fit(num_weak_learners=100)
