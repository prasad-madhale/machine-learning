import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DualPerceptron:

    def __init__(self, kernel, path):
        self.kernel = kernel
        self.x, self.y = DualPerceptron.get_data(path)

    @staticmethod
    def get_data(path):
        df = pd.read_csv(path, delim_whitespace=True, header=None).values
        data = df[:, :-1]
        label = df[:, -1]

        # normalize data
        data = DualPerceptron.normalize(data)

        return data, label

    @staticmethod
    def normalize(data):
        std = StandardScaler()
        std.fit(data)
        std.transform(data)
        return data

    def mistakes(self, w):
        temp = np.sum(self.k_vals(self.x) * w, axis=1)
        result = (self.y * temp) <= 0
        result = result.reshape(result.shape[0])
        return result

    def k_vals(self, test):
        if self.kernel == 'gaussian':
            vals = []

            for test_point in test:
                value = self.gaussian(test_point)
                vals.append(value)
            vals = np.array(vals)
        elif self.kernel == 'linear':
            vals = self.x.dot(test.T)

        return vals

    def fit(self):
        w = np.zeros(self.x.shape[0])
        x_mistake = self.x[self.mistakes(w)]

        iters = 0

        while len(x_mistake) > 0:
            mistake_indices = self.mistakes(w)
            w[mistake_indices] += self.y[mistake_indices].flatten()
            x_mistake = self.x[mistake_indices]

            print('Iteration: {} | Misclassified Points: {}'.format(iters, len(x_mistake)))

            iters += 1

    def gaussian(self, point, sigma=2):
        return np.exp(-1 * (np.sum((self.x - point)**2, axis=1) / (sigma**2)))


# EXECUTION
# dp = DualPerceptron(kernel='gaussian', path='./data/perceptronData.txt')
# dp = DualPerceptron(kernel='linear', path='./data/twoSpirals.txt')
# dp = DualPerceptron(kernel='gaussian', path='./data/twoSpirals.txt')

dp.fit()
