import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, auc, roc_curve


class AdaBoost:

    def __init__(self):
        self.train_data, self.train_labels, self.test_data, self.test_labels = AdaBoost.get_data()
        self.data = AdaBoost.normalize(self.data.astype(np.float))

    @staticmethod
    def get_data():
        test_data = pd.read_csv('./data/spam_polluted/test_feature.txt', sep=' ')
        test_label = pd.read_csv('./data/spam_polluted/test_label.txt', sep='\n')
        train_data = pd.read_csv('./data/spam_polluted/train_feature.txt', sep=' ')
        train_label = pd.read_csv('./data/spam_polluted/train_label.txt', sep='\n')

        return train_data, train_label, test_data, test_label

    @staticmethod
    def normalize(dataset):
        return preprocessing.minmax_scale(dataset, feature_range=(0, 1))

    def fit(self, num_weak_learners=100):
        train_data, test_data, train_label, test_label = train_test_split(self.data, self.labels, test_size=0.25)

        # train for the training data
        models, summary, features = AdaBoost.train(train_data, train_label, test_data, test_label, num_weak_learners)

        return features

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

        features = []

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

            features.append(model.feature)

            train_error, _ = AdaBoost.get_error(models, train_data, train_label)
            test_error, auc_score = AdaBoost.get_error(models, test_data, test_label)

            # add to summary all the values needed for plotting
            summary[ep] = [model.error, train_error, test_error, auc_score]

            print('Round: {}  |  Feature: {}  |  Threshold: {}  |  Round_error: {}  |  Train_error: {}  |  '
                  'Test_error: {}  |  AUC: {}'.format(ep+1, model.feature, model.threshold, model.error, train_error,
                                                      test_error, auc_score), file=f)

        return models, summary, features

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


with open('./logs/out_polluted.txt', 'w') as f:
    ada_boost = AdaBoost()
    print(ada_boost.train_data.shape, ada_boost.train_labels.shape, ada_boost.test_data.shape,
          ada_boost.test_labels.shape)

print('Done Training!')
