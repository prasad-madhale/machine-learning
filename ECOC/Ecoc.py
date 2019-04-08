
import numpy as np
import sys
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, auc, roc_curve
import random
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy.spatial import distance


def get_data(num_datapoints, num_features, path):
    file_op = open(path, "r")

    data = np.empty(shape=(num_datapoints, num_features))
    label = np.empty(shape=(num_datapoints,))

    for line_num, line in enumerate(file_op):

        # split all lines with ' ' as delimiter
        splitter = line.split(sep=' ')

        # remove last element which is a new line '\n'
        splitter = splitter[:-1]

        # add label
        label[line_num] = splitter[0]

        for ele_num in range(1, len(splitter)):
            splitted = splitter[ele_num].split(':')
            splitted = np.array(splitted, dtype=np.float)
            data[line_num][int(splitted[0])] = splitted[1]

    file_op.close()

    return data, label


# OPTIMAL-THRESHOLD ADA BOOST
class AdaBoost:

    @staticmethod
    def fit(num_weak_learners, data, labels):
        train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=0.20)

        # train for the training data
        models, summary = AdaBoost.train(train_data, train_label, test_data, test_label, num_weak_learners)

        error, acc = AdaBoost.get_error(models, data, labels)

        print('Training Accuracy(for individual Adaboost): {}'.format(acc))

        return models

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

            # print('Round: {}  |  Feature: {}  |  Threshold: {}  |  Round_error: {}  |  Train_error: {}  |  '
            #       'Test_error: {}  |  AUC: {}'.format(ep+1, model.feature, model.threshold, model.error, train_error,
            #                                           test_error, auc_score))

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
    def get_midpoints(data):
        thresholds = []

        for i in range(len(data)):
            thres = data[i]
            midpoints = (thres[1:] + thres[:-1]) / 2

            # # append min and max of that field's threshold midpoints
            # min_max = np.array([min(train_data[:, i]), max(train_data[:, i])])

            # np.concatenate((midpoints, min_max))
            thresholds.append(midpoints)

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


class ECOC:

    def __init__(self, train_data, train_label, test_data, test_label, num_learners, coding_matrix):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.num_weak_learners = num_learners
        self.c_matrix = coding_matrix

    def train(self):

        train_labels = []

        # create inputs that will go into the multiprocessing code
        for feature_num in range(self.c_matrix.shape[1]):

            # creates a set of what classes should be classified as 1 for this particular adaboost
            feature_set = set()

            # get values from the coding matrix for this feature
            feature_col = self.c_matrix[:, feature_num]

            # add values to the feature set for which value in the coding matrix is 1
            for label_num, val in enumerate(feature_col):
                if val == 1:
                    feature_set.add(label_num)

            # fork copy of train_label as we will be modifying the training labels
            train_label = self.train_label.copy()

            # modify the labels to make it a 0/1 classification problem for adaboost
            train_label = ECOC.modify_labels(train_label, feature_set)

            # create an array of inputs for multiprocessing
            train_labels.append(train_label)

        with Pool(cpu_count()) as pool:
            func = partial(AdaBoost.fit, self.num_weak_learners, train_data)
            models = pool.map(func=func, iterable=train_labels)

            pool.close()
            pool.join()

        # get train accuracy for final model
        train_acc = ECOC.get_accuracy(models, self.train_data, self.train_label, self.c_matrix)

        print('Final Training Accuracy: {}'.format(train_acc))

        # get test accuracy for final model
        test_acc = ECOC.get_accuracy(models, self.test_data, self.test_label, self.c_matrix)

        print('Final Test Accuracy: {}'.format(test_acc))

    @staticmethod
    def get_accuracy(models, data, labels, code_matrix):
        # get prediction on data
        pred = ECOC.predict(models, data)

        # map the code predictions to actual label classes
        pred = ECOC.map_pred_to_classes(pred, code_matrix)

        # get accuracy for predictions given truth
        return accuracy_score(pred, labels)


    @staticmethod
    def predict(models, data):
        final_matrix = []

        for model in models:
            # get prediction
            prediction = WeakLearner.get_prediction(model, data)

            # convert all -1s to 0s
            prediction[prediction < 0] = 0

            # append to return
            final_matrix.append(prediction)

        final_matrix = np.array(final_matrix)

        return  np.transpose(final_matrix)

    @staticmethod
    def map_pred_to_classes(pred_in_code, code_matrix):
        predict = np.empty(shape=(len(pred_in_code,)))

        # iterate through the final matrix
        for row_num, row in enumerate(pred_in_code):

            min_dist = float('inf')

            # find the code which is closest to the prediction we obtained
            # by using hamming distance as the metric
            for i, c_row in enumerate(code_matrix):
                dist = distance.hamming(row, c_row)

                # store the label class with least hamming distance
                if dist < min_dist:
                    min_dist = dist
                    label = i

            predict[row_num] = label

        return predict

    @staticmethod
    def modify_labels(label, feature_set):

        for ele in range(len(label)):
            if label[ele] in feature_set:
                label[ele] = 1
            else:
                label[ele] = -1

        return label


def normalize(data):
    return preprocessing.minmax_scale(data, feature_range=(0, 1))


# EXECUTION

# fix seed for all random operations
np.random.seed(20)
random.seed(20)

# constants
NUM_TRAIN_DATAPOINTS = 11314
NUM_TEST_DATAPOINTS = 7532
NUM_FEATURES = 1754

# printing numpy array without annoying truncation
np.set_printoptions(threshold=sys.maxsize)

# get training data
train_data, train_label = get_data(NUM_TRAIN_DATAPOINTS, NUM_FEATURES, path="./data/8newsgroup/train.trec/feature_matrix.txt")
train_data = normalize(data=train_data)

# get testing data
test_data, test_label = get_data(NUM_TEST_DATAPOINTS, NUM_FEATURES, path="./data/8newsgroup/test.trec/feature_matrix.txt")
test_data = normalize(data=test_data)

# generate coding matrix for ECOC procedure
coding_matrix = np.array([[1,0,1,0,0,1,1,0,0,1,1,0,1,0,1,1,1,1,1,0],
                        [1,0,0,0,0,1,0,1,0,0,1,1,0,0,1,1,0,1,0,1],
                        [1,1,0,0,0,0,1,0,1,0,0,1,1,0,0,1,1,0,1,0],
                        [1,1,1,0,0,0,0,1,0,1,0,0,1,1,0,0,1,1,0,1],
                        [1,1,1,1,0,0,0,0,1,0,1,0,0,1,1,0,0,1,1,0],
                        [1,1,1,1,1,0,0,0,0,1,0,1,0,0,1,1,0,0,1,1],
                        [0,1,1,1,1,1,0,0,0,0,1,0,1,0,0,1,1,0,0,1],
                        [1,0,1,1,1,1,1,0,0,0,0,1,0,1,0,0,1,1,0,0]])

ecoc = ECOC(train_data, train_label, test_data, test_label, 10, coding_matrix)
ecoc.train()