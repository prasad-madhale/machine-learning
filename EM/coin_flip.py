import numpy as np
from collections import Counter


class CoinFlipEM:

    def __init__(self, m=1000, k=10, p=0.7, r=0.2, pi=0.5):
        self.p = p
        self.r = r
        self.data, self.label = CoinFlipEM.generate_data(m, k, p, r, pi)

    @staticmethod
    def generate_data(m, k, p, r, pi):
        data = np.zeros((m, k))
        label = np.zeros(m)
        probabilities = {1: p, 2: r}

        # m times pick coins based on pi probability
        for i in range(m):
            rand = np.random.uniform(0, 1)

            if pi > rand:
                coin = 1
            else:
                coin = 2

            label[i] = coin
            coin_prob = probabilities[coin]

            for j in range(k):
                rand_coin = np.random.uniform(0, 1)

                if coin_prob > rand_coin:
                    data[i][j] = 1
                else:
                    data[i][j] = 0

        return data, label

    def train(self, max_steps=50):

        # random parameters to begin with
        pi_1 = np.random.uniform(0, 1)
        pi_2 = (1 - pi_1)
        pi_s = [pi_1, pi_2]

        for itr in range(max_steps):
            expects = self.e_step(pi_s)
            pi_s = self.m_step(pi_s, expects)

        print('Expected: {}   |   {}'.format(self.p, self.r))
        print('Result: {}   |   {}'.format(pi_s[0], pi_s[1]))

    def e_step(self, pi_s):

        heads_1 = heads_2 = 0
        tails_1 = tails_2 = 0

        for pt in self.data:
            count = Counter(pt)

            likelihood_1 = self.likelihood(pi_s[0], count)
            likelihood_2 = self.likelihood(pi_s[1], count)

            total_likelihood = (likelihood_1 + likelihood_2)
            prob_1 = likelihood_1 / total_likelihood
            prob_2 = likelihood_2 / total_likelihood

            heads_1 += prob_1 * count[1]
            tails_1 += prob_1 * count[0]

            heads_2 += prob_2 * count[1]
            tails_2 += prob_2 * count[0]

        return [[heads_1, tails_1], [heads_2, tails_2]]

    def likelihood(self, pi, count):
        return pi**(count[1]) * (1-pi)**(count[0])

    def m_step(self, pi_s, expects):
        pi_s[0] = expects[0][0] / (expects[0][0] + expects[0][1])
        pi_s[1] = expects[1][0] / (expects[1][0] + expects[1][1])

        return pi_s


np.random.seed(2)
coinEM = CoinFlipEM()
coinEM.train()

