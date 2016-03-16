import math
from time import time

__author__ = 'vks'


class Node:
    feature = -1
    key = -1
    left = None
    type = -1  # this field is assigned when node is a leaf; the type with the highest probability is selected.
    # In case when multiple types have it, the first one is selected.
    right = None
    depth = 0


class Tree:
    def __init__(self, max_depth=-1, cut_edges=False, gain_func='gini', threshold=1e-2):
        self._max_depth = max_depth
        self._cut_edges = cut_edges
        self._data = []
        self._categories = []
        self._cat_right = []
        self._cat_left = []
        self._max_category = -1
        self._root = None
        self._threshold = threshold
        if gain_func == 'gini':
            self._gain_func = self._gini
        elif gain_func == 'entropy':
            self._gain_func = self._entropy

    def fit(self, data, categories):
        self._data = data
        self._categories = categories
        for i in categories:
            self._max_category = max(self._max_category, i)
        self._max_category += 1
        self._cat_left = [0 for i in range(self._max_category)]
        self._cat_right = [0 for i in range(self._max_category)]
        self._root = Node()
        self._build_tree(data, categories, self._root)

    def predict(self, element):
        if self._root is None:
            return None
        current = self._root
        while current.left is not None and current.right is not None:
            if element[current.feature] <= current.key:
                current = current.left
            else:
                current = current.right
        return current.type

    def set_max_depth(self, val):
        self._max_depth = val

    def perform_edges_cut(self, val):
        self._cut_edges = val

    def _build_tree(self, data, categories, node):
        feature = -1
        key = -1
        gain = 0

        if len(data) > 1:
            size = len(categories)
            for i in range(len(data[0])):
                packed_data = sorted([(data[j][i], categories[j]) for j in range(len(categories))], key=lambda x: x[0])
                cat_right = [j[1] for j in packed_data]
                overall_gain = self._calculate_criterion('init', cat_right, size)
                ind = 0
                while len(cat_right) > 0:
                    diff = [cat_right.pop()]
                    ind += 1
                    while len(cat_right) > 0 and packed_data[ind][0] == packed_data[ind - 1][0]:
                        diff.append(cat_right.pop())
                        ind += 1
                    test_gain = overall_gain - \
                                ind / size * self._calculate_criterion('left', diff, ind) - \
                                len(cat_right) / size * self._calculate_criterion('right', diff, len(cat_right))
                    if test_gain > gain:
                        gain = test_gain
                        feature = i
                        key = packed_data[ind - 1][0]

        if (self._cut_edges and gain < self._threshold) or (self._max_depth == node.depth) or (gain < 1e-6):
            node.type = self._major_type(categories)
            return

        data_left = []
        data_right = []
        categories_left = []
        categories_right = []
        for i in range(len(data)):
            if data[i][feature] <= key:
                data_left.append(data[i])
                categories_left.append(categories[i])
            else:
                data_right.append(data[i])
                categories_right.append(categories[i])
        node.left = Node()
        node.right = Node()
        node.left.depth = node.depth + 1
        node.right.depth = node.depth + 1
        node.key = key
        node.feature = feature
        self._build_tree(data_left, categories_left, node.left)
        self._build_tree(data_right, categories_right, node.right)

    def _entropy(self, val):
        return val * math.log(val, 2)

    def _gini(self, val):
        return val ** 2

    def _calculate_criterion(self, mode, diff, size):
        if size == 0:
            return 1
        ans = 1
        if mode == 'right':
            for x in diff:
                self._cat_right[x] -= 1
            for v in self._cat_right:
                ans -= self._gain_func(v / size)
        elif mode == 'left':
            for x in diff:
                self._cat_left[x] += 1
            for v in self._cat_left:
                ans -= self._gain_func(v / size)
        elif mode == 'init':
            self._cat_left = [0 for i in range(self._max_category)]
            self._cat_right = [0 for i in range(self._max_category)]
            for x in diff:
                self._cat_right[x] += 1
            for v in self._cat_right:
                ans -= self._gain_func(v / size)
        return ans

    def _major_type(self, categories):
        cat = {}
        for x in categories:
            if x not in cat:
                cat[x] = 1
            else:
                cat[x] += 1
        ans = 0
        val = 0
        for k, v in cat.items():
            if v > val:
                val = v
                ans = k
        return ans
