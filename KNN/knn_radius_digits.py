import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.metrics import balanced_accuracy_score
from tensorflow.examples.tutorials.mnist import input_data


def normalize(data, train_size):
    std = StandardScaler()
    std.fit(data)
    std.transform(data)

    train_data = data[:train_size]
    test_data = data[train_size:]

    return train_data, test_data


class KNN:

    def __init__(self, radius, kernel):
        self.radius = radius
        self.kernel = kernel
        self.x = None
        self.y = None

    def train(self, x_train, y_train):

        # store the train data and labels
        self.x = x_train
        self.y = y_train

    def get_distance(self, point):
        return np.linalg.norm(self.x - point, axis=1)

    def cosine_kernel(self, point):
        return 1 - self.x.dot(point) / (np.sum(self.x**2, axis=1)**0.5 * (np.sum(point**2)**0.5))

    def k_closest_points(self, point):
        if self.kernel == 'euclidean':
            distances = self.get_distance(point)
        elif self.kernel == 'cosine':
            distances = self.cosine_kernel(point)
        else:
            raise Exception('Please enter valid Kernel choice')

        return distances < self.radius

    def predict(self, x_test):
        predictions = []

        for pt in x_test:
            closest_indices = self.k_closest_points(pt)
            closest_pt_labels = self.y[closest_indices]

            # if no points within the radius we return majority label
            try:
                pt_prediction = stats.mode(closest_pt_labels)[0][0]
            except:
                pt_prediction = stats.mode(self.y)[0][0]

            predictions.append(pt_prediction)

        return predictions

    def train_test_accuracy(self, x_test, y_test):

        train_prediction = self.predict(self.x)
        train_acc = balanced_accuracy_score(self.y, train_prediction)

        test_prediction = self.predict(x_test)
        test_acc = balanced_accuracy_score(y_test, test_prediction)

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
RADIUS = 0.31

# load mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

x_train = np.vstack([img.reshape(-1, ) for img in mnist.train.images])
y_train = mnist.train.labels

x_test = np.vstack([img.reshape(-1, ) for img in mnist.test.images])
y_test = mnist.test.labels

# reduce the training data set to include only 20% of all each label data
x_train, y_train = reduce_train(x_train, y_train)

# normalize data
full_data = np.concatenate([x_train, x_test])
x_train, x_test = normalize(full_data, x_train.shape[0])

with open('./logs/out_knn_radius_digits', 'w') as file_op:
    knn = KNN(RADIUS, kernel='cosine')
    knn.train(x_train, y_train)

    train_accs, test_accs = knn.train_test_accuracy(x_test, y_test)

    print('For kernel = {}'.format(knn.kernel), file=file_op)
    print('For radius = {}'.format(knn.radius), file=file_op)
    print('Train accuracy: {}'.format(train_accs), file=file_op)

    print('Test accuracy: {}'.format(test_accs), file=file_op)

