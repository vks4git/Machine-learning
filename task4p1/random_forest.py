from tree import Tree
from random import uniform

__author__ = 'vks'


class RandomForest:
    def __init__(self, size=10, features=0.25):
        self._size = size
        self._features = features
        self._trees = []
        self._masks = []

    def fit(self, data, types):
        self._trees = []
        self._masks = []
        features = max(1, round(self._features * len(data[0])))
        for iter in range(self._size):
            self._trees.append(Tree())
            mask = [False for j in range(len(data[0]))]
            for j in range(features):
                mask[j] = True
            for j in range(len(mask)):
                ind = round(uniform(0, len(mask) - 1))
                tmp = mask[j]
                mask[j] = mask[ind]
                mask[ind] = tmp
            self._masks.append(mask)
            data_x = []
            data_y = []
            for j in range(len(data)):
                ind = round(uniform(0, len(data) - 1))
                vec = []
                for k in range(len(mask)):
                    if mask[k]:
                        vec.append(data[ind][k])
                data_x.append(vec)
                data_y.append(types[ind])
            self._trees[iter].fit(data_x, data_y)

    def predict(self, element):
        votes = {}
        for i in range(self._size):
            vec = []
            for j in range(len(element)):
                if self._masks[i][j]:
                    vec.append(element[j])
            vote = self._trees[i].predict(vec)
            if vote not in votes.keys():
                votes[vote] = 1
            else:
                votes[vote] += 1
        max_votes = 0
        ans = 0
        for k, v in votes.items():
            if v > max_votes:
                max_votes = v
                ans = k
        return ans
