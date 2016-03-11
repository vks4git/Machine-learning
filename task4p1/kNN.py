from metrics import *

__author__ = 'vks'


class Naive_kNN:
    def __init__(self, k=10, metrics=euclides):
        self._dataset = []
        self._metrics = metrics
        self._k = k

    def set_k(self, k):
        self._k = k

    def learn(self, data_x, data_y):
        self._dataset = [data_x, data_y]

    def classify(self, element):
        distances = []
        for i in range(len(self._dataset[0])):
            distances.append((self._metrics(self._dataset[0][i], element), self._dataset[1][i]))
        distances.sort()
        types = {}
        for i in range(self._k):
            if distances[i][1] not in types.keys():
                types[distances[i][1]] = 0
            types[distances[i][1]] += 1
        max_type = -1
        max_val = -1
        for k, v in types.items():
            if v > max_val:
                max_val = v
                max_type = k
        return max_type


class Weighted_kNN:
    def __init__(self, k=10, metrics=euclides):
        self._dataset = []
        self._metrics = metrics
        self._k = k

    def fit(self, data_x, data_y):
        self._dataset = [data_x, data_y]
        data_size = len(self._dataset[0])
        vec_size = len(self._dataset[0][0])
        max_vec = [0 for i in range(vec_size)]
        for i in self._dataset[0]:
            for j in range(len(i)):
                max_vec[j] = max(max_vec[j], i[j])
        for i in range(data_size):
            for j in range(vec_size):
                self._dataset[0][j] /= max_vec[j]

    def set_k(self, k):
        self._k = k

    def predict(self, element):
        distances = []
        for i in range(len(self._dataset[0])):
            distances.append((self._metrics(self._dataset[0][i], element), self._dataset[1][i]))
        distances.sort()
        types = {}
        for i in range(self._k):
            if distances[i][1] not in types.keys():
                types[distances[i][1]] = 0
            types[distances[i][1]] += 1
        max_type = -1
        max_val = -1
        for k, v in types.items():
            if v > max_val:
                max_val = v
                max_type = k
        return max_type
