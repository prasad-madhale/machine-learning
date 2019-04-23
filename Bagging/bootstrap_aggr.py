
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from multiprocessing import Pool, cpu_count
from functools import partial


class DecisionTree:

    def __init__(self, max_depth, train_set, test_set):
        self.max_depth = max_depth
        self.train_data = train_set
        self.test_data = test_set
        self.model = None

    @staticmethod
    def get_thresholds(dataset, feature):
        return dataset[feature].unique()

    @staticmethod
    def get_best_split(dataset):

        best_feature = None
        best_threshold = None
        max_info_gain = 0

        gini_before = DecisionTree.gini(dataset)

        for feature in dataset.columns[:-1]:

            thresholds = DecisionTree.get_thresholds(dataset, feature)

            for threshold in thresholds:
                left, right = DecisionTree.split_data(dataset, feature, threshold)

                if len(left) == 0 or len(right) == 0:
                    continue

                left_gini = DecisionTree.gini(left)
                right_gini = DecisionTree.gini(right)

                w = len(left) / len(dataset)
                gini_after = (w * left_gini) + ((1 - w) * right_gini)

                info_gain = gini_before - gini_after

                if max_info_gain <= info_gain:
                    max_info_gain = info_gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, max_info_gain

    @staticmethod
    def get_value_count(dataset):
        return dataset.spam_label.value_counts()

    @staticmethod
    def gini(dataset):
        counts = DecisionTree.get_value_count(dataset)

        imp = 1
        for x in counts:
            prob = x / len(dataset)
            imp -= prob ** 2

        return imp

    @staticmethod
    def split_data(dataset, feature, threshold):

        left = dataset[dataset[feature] < threshold]
        right = dataset[dataset[feature] >= threshold]

        return left, right

    def fit(self):
        self.model = DecisionTree.build_tree(self.max_depth, 0, self.train_data)

    def test(self):
        return DecisionTree.test_model(self.model, self.test_data)

    @staticmethod
    def build_tree(max_depth, depth, dataset):

        best_feature, best_threshold, info_gain = DecisionTree.get_best_split(dataset)

        if info_gain == 0 or depth >= max_depth:
            return Terminal(dataset)

        left_data, right_data = DecisionTree.split_data(dataset, best_feature, best_threshold)

        left_node = DecisionTree.build_tree(max_depth, depth + 1, left_data)
        right_node = DecisionTree.build_tree(max_depth, depth + 1, right_data)

        return Node(best_feature, best_threshold, left_node, right_node)

    @staticmethod
    def predict(root, entry):
        if isinstance(root, Terminal):
            return root.predict()

        if entry[root.feature] < root.threshold:
            result = DecisionTree.predict(root.left_node, entry)
        else:
            result = DecisionTree.predict(root.right_node, entry)

        return result

    @staticmethod
    def test_model(model, test_data):
        predictions = []

        # build predictions
        for i, entry in test_data.iterrows():
            predictions.append(DecisionTree.predict(model, entry))

        # get actual labels
        test_labels = test_data['spam_label']

        return accuracy_score(test_labels, predictions), predictions


class Node:

    def __init__(self, feature, threshold, left_node, right_node):
        self.feature = feature
        self.threshold = threshold
        self.left_node = left_node
        self.right_node = right_node


class Terminal:

    def __init__(self, dataset):
        self.prediction = dataset.spam_label.mode()[0]

    def predict(self):
        return self.prediction


def get_data(column_names):
    data_frame = pd.read_csv('./data/spambase.txt', sep=',')
    data_frame.columns = column_names
    return data_frame


def normalize(dataset):
    # normalize everything apart from the labels
    cols = dataset.columns[:-1]
    dataset[cols] = dataset[cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    return dataset


class Bagging:

    def __init__(self, t, n, train_set, test_set, max_depth):
        self.times = t
        self.n = n
        self.train_data = train_set
        self.test_data = test_set
        self.max_depth = max_depth
        self.models = None

    def fit(self):
        inputs = []

        # make multiple random sampled data bags with replacement
        for t in range(self.times):

            # pick n random indices to form bags
            # numpy random choice assumes a uniform distribution unless mentioned explicitly
            indices = np.random.choice(len(self.train_data), self.n)

            # random sample dataset
            bag = self.train_data.iloc[indices]

            inputs.append(bag)

        with Pool(cpu_count()) as pool:
            func = partial(DecisionTree.build_tree, self.max_depth, 0)
            models = pool.map(func, inputs)

        self.models = models

    def test(self, option):

        if option == 'train':
            test_data = self.train_data
        elif option == 'test':
            test_data = self.test_data
        else:
            raise Exception('Provide valid input')

        all_preds = []

        for model in self.models:
            acc, predictions = DecisionTree.test_model(model, test_data)
            all_preds.append(predictions)

        # convert to numpy array
        all_preds = np.array(all_preds)

        print(all_preds.shape)

        final_preds = []

        for i in range(len(test_data)):
            arr = all_preds[:, i]
            counts = np.bincount(arr)
            final_preds.append(np.argmax(counts))

        acc = accuracy_score(test_data['spam_label'], final_preds)
        return acc, final_preds


# EXECUTION

# set seed for random operations
np.random.seed(2)

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
                'char_freq_(', 'char_freq_[', 'char_freq_!','char_freq_$', 'char_freq_#', 'capital_run_length_average',
                'capital_run_length_longest', 'capital_run_length_total', 'spam_label']

# get data from the txt file
df = get_data(column_names)

# normalize the data
df = normalize(df)

# shuffle data
df = df.sample(frac=1)

# split the data into test and train set
test_size = int(0.20 * len(df))
test_data = df[:test_size]
train_data = df[test_size:]

# lets use 70% data as the n for bagging
train_size = int(0.70 * len(train_data))

boot_agr = Bagging(t=50, n=train_size, train_set=train_data, test_set=test_data, max_depth=2)
boot_agr.fit()

# get testing accuracy
test_acc, _, _ = boot_agr.test('test')

print('Test Accuracy: {}'.format(test_acc))
