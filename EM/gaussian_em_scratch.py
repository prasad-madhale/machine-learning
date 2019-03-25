#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 20:29:49 2019

@author: prasad
"""
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


class EM:
    
    def __init__(self, k):
        self.k = k
        self.data = EM.get_data()
        self.w = np.asmatrix(np.empty(self.data.shape), dtype=float)
        self.phi = np.ones(self.data.shape[1])/2
        self.model = EM.init_model(k)

    @staticmethod
    def init_model(k):
        return {'mu': np.asmatrix(np.random.random((k, k))), 'sig': np.array([np.identity(k) for _ in range(k)])}

    @staticmethod
    def get_data():
        df = pd.read_csv('./data/2gaussian.txt', sep=' ')
        return df.values
    
    def e_step(self):
        for i in range(len(self.data)):
            den = 0
            
            for j in range(self.k):
                prob = multivariate_normal.pdf(self.data[i], self.model['mu'][j].A1, self.model['sig'][j]) \
                       * self.phi[j]
                
                den += prob
                self.w[i, j] = prob
            
            self.w[i] /= den

    def maximization(self):
        
        for i in range(self.k):
            sum_w = self.w[:, i].sum()
            self.phi[i] = sum_w / len(self.w)
            new_mu = np.zeros(self.k)
            new_sigma = np.zeros((self.k, self.k))
            
            for j in range(len(self.data)):
                new_mu += (self.data[j] * self.w[j, i])
                diff = np.asmatrix(np.array(self.data[j] - self.model['mu'][i]))
                new_sigma += (self.w[j, i] * np.multiply(np.transpose(diff), diff))

            self.model['mu'][i] = new_mu / sum_w
            self.model['sig'][i] = new_sigma / sum_w

    def log_likelihood(self):
        ans = 0
        
        for i in range(len(self.data)):
            log = 0
            
            for j in range(self.k):
                log += multivariate_normal.pdf(self.data[i], self.model['mu'][j].A1, self.model['sig'][j]) * self.phi[j]
            
            ans += np.log(log)
        
        return ans
    
    def train(self, threshold=1e-4):
        diff = float('inf')
        itr = 0
        
        while diff > threshold:
            # calculate previous set log likelihood
            prev_ll = self.log_likelihood()
            
            # perform e step
            self.e_step()

            # perform maximization step
            self.maximization()
            
            # recalculate log_likelihood
            new_ll = self.log_likelihood()
            
            diff = new_ll - prev_ll
            
            itr += 1
            
            print('Iter: {}, LogLikeHood Difference: {}'.format(itr, diff))
            
        print(self.model)


gaussEM = EM(2)
gaussEM.train()
