from random import random, uniform
import numpy
from math import exp, log
from time import time


class AdaBoost:
    def __init__(self, k):
        self._func_count = k
        self._weights = None
        self._alpha = numpy.ones(k)
        self._masks = None
        self._data_x = None
        self._data_y = None

    def fit(self, data_x, data_y):
        self._data_x = data_x
        self._data_y = data_y
        self._weights = numpy.array([1 / len(data_y)] * len(data_y))
        self._masks = numpy.array([self._gen_haar_mask(784) for i in range(self._func_count)])
        # start = time()
        for i in range(self._func_count):
            # print("Iteration %s: %s" % (i, time() - start))
            err = 0
            for j in range(len(self._data_y)):
                err += self._weights[j] * (1 if self._sign(numpy.dot(self._masks[i], data_x[j])) == data_y[j] else 0)
            if err > 0.5:
                self._masks[i] = numpy.multiply(self._masks[i], -1)
                err = 1 - err
            err = max(err, 1e-10)  # to prevent evaluating of log(0)
            self._alpha[i] = 0.5 * log((1 - err) / (1 + err))
            for j in range(len(data_y)):
                self._weights[j] *= exp(-self._alpha[i] * data_y[j] * self._sign(numpy.dot(self._masks[i], data_x[j])))
            w0 = numpy.sum(self._weights)
            self._weights = numpy.multiply(self._weights, 1 / w0)

    def predict(self, x):
        ans = 0
        for i in range(self._func_count):
            ans += numpy.dot(self._masks[i], x) * self._alpha[i]
        return self._sign(ans)

    def _gen_haar_mask(self, size):
        return [(1 if random() > 0.5 else -1) for i in range(size)]

    def _sign(self, x):
        if x < 0:
            return -1
        return 1


class AdaBoostMulticlass:
    def __init__(self, k):
        self._func_count = k
        self._classificators = []
        self._classes = 0

    def fit(self, data_x, data_y):
        dict = {}
        for y in data_y:
            if y not in dict:
                dict[y] = 1
        self._classes = len(dict)
        self._classificators = [AdaBoost(self._func_count) for i in range(self._classes)]
        for i in range(self._classes):
            new_y = numpy.array([1 if i == y else -1 for y in data_y])
            self._classificators[i].fit(data_x, new_y)

    def predict(self, x):
        votes = [0] * self._classes
        for i in range(self._classes):
            votes[i] += self._classificators[i].predict(x)
        ans = int(uniform(0, self._classes))
        argmax = votes[ans]
        for i in range(self._classes):
            if votes[i] > argmax:
                argmax = votes[i]
                ans = i
        return ans
