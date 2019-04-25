import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score
from tensorflow.examples.tutorials.mnist import input_data


def get_data(column_names):
    data_frame = pd.read_csv('./data/spambase.txt', header=None, sep=',', dtype=np.float)
    data_frame.columns = column_names

    label = data_frame['spam_label'].values
    data = data_frame.drop(labels=['spam_label'], axis=1).values

    return data, label


def normalize(data):
    std = StandardScaler()
    std.fit(data)
    std.transform(data)
    return data


class KNN:

    def __init__(self, kernel):
        self.kernel = kernel
        self.x = None
        self.y = None
        self.pr = None

    def train(self, x_train, y_train):

        # store the train data and labels
        self.x = x_train
        self.y = y_train
        self.pr = self.priors()

    def priors(self):
        classes = np.unique(self.y)

        priors = []

        for label in classes:
            pts = self.y[self.y == label]
            prior = len(pts) / len(self.y)
            priors.append(prior)

        return np.array(priors)

    def get_distance(self, point):
        return np.linalg.norm(self.x - point, axis=1)

    def cosine_kernel(self, point):
        return 1 - self.x.dot(point) / (np.sum(self.x**2, axis=1)**0.5 * (np.sum(point**2)**0.5))

    def gaussian(self, point):
        return np.exp(-1 * np.sum((self.x - point)**2, axis=1))

    def poly_kernel(self, test):
        return (test.dot(self.x.T) ** 2) + 1

    def points(self, test):
        if self.kernel == 'gaussian':
            distances = []

            for entry in test:
                distances.append(self.gaussian(entry))

        elif self.kernel == 'poly':
            distances = self.poly_kernel(test)
        else:
            raise Exception('Please enter valid Kernel choice')

        return np.array(distances)

    def get_probs(self, test):
        distances = self.points(test)
        num_labels = np.unique(self.y)
        label_probs = []

        for i, k in enumerate(distances):
            probs = []

            for label in num_labels:
                p = np.sum(distances[i][self.y == label]) / np.sum(self.y == label)
                probs.append(p)

            label_probs.append(probs)

        return np.array(label_probs)

    def predict(self, x_test):
        predictions = self.get_probs(x_test) * self.pr
        return np.argmax(predictions, axis=1)

    def train_test_accuracy(self, x_test, y_test):

        train_prediction = self.predict(self.x)
        train_acc = accuracy_score(self.y, train_prediction)

        test_prediction = self.predict(x_test)
        test_acc = accuracy_score(y_test, test_prediction)

        return train_acc, test_acc


def reduce_train(x_train, y_train, data_percent=0.20):
    new_train = np.empty(shape=(1, 784))
    new_labels = np.array([])

    for label in range(10):
        indices = np.where(y_train == label)[0]

        # number of values to pick
        new_size = int(data_percent * len(indices))

        # 20% values
        picks = np.random.choice(indices, new_size)

        label_reduced = x_train[picks]

        new_train = np.concatenate([new_train, label_reduced])

        r_labels = y_train[picks]

        new_labels = np.concatenate([new_labels, r_labels])

    return np.array(new_train[1:]), np.array(new_labels)


# EXECUTION
np.random.seed(2)

# load mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

x_train = np.vstack([img.reshape(-1, ) for img in mnist.train.images])
y_train = mnist.train.labels

x_test = np.vstack([img.reshape(-1, ) for img in mnist.test.images])
y_test = mnist.test.labels

# reduce the training data set to include only 20% of all each label data
x_train, y_train = reduce_train(x_train, y_train)

with open('./logs/out_prob_digits', 'w') as file_op:
    knn = KNN(kernel='gaussian')
    knn.train(x_train, y_train)

    train_accs, test_accs = knn.train_test_accuracy(x_test, y_test)

    print('For kernel = {}'.format(knn.kernel), file=file_op)
    print('Train accuracy: {}'.format(train_accs), file=file_op)
    print('Test accuracy: {}'.format(test_accs), file=file_op)

print('Done!')
