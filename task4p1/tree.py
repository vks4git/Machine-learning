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
        self._cat_right = []
        self._cat_left = []
        self._max_category = -1
        self._root = None
        self._threshold = threshold
        self._calls = 0  # todo:remove this
        if gain_func == 'gini':
            self._gain_func = self._gini
        elif gain_func == 'entropy':
            self._gain_func = self._entropy

    def fit(self, data, categories):
        for i in categories:
            self._max_category = max(self._max_category, i)
        self._max_category += 1
        self._cat_left = [0] * self._max_category
        self._cat_right = [0] * self._max_category
        self._root = Node()
        packed_data = [(data[j], categories[j]) for j in range(len(categories))]
        self._build_tree(packed_data, self._root)
        print(self._calls)

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

    def _build_tree(self, data, node):
        self._calls += 1
        feature = -1
        key = -1
        gain = 0
        left_size = 0
        right_size = 0
        if len(data) > 1:
            size = len(data)
            for i in range(len(data[0][0])):
                data.sort(key=lambda elem: elem[0][i])
                cat_right = [j[1] for j in data]
                overall_gain = self._calculate_criterion('init', cat_right, size)
                ind = 0
                while len(cat_right) > 0:
                    diff = [cat_right.pop()]
                    ind += 1
                    while len(cat_right) > 0 and data[ind][0][i] == data[ind - 1][0][i]:
                        diff.append(cat_right.pop())
                        ind += 1
                    test_gain = overall_gain - \
                                ind / size * self._calculate_criterion('left', diff, ind) - \
                                len(cat_right) / size * self._calculate_criterion('right', diff, len(cat_right))
                    if test_gain > gain:
                        gain = test_gain
                        feature = i
                        key = data[ind - 1][0][i]
                        left_size = ind
                        right_size = len(data) - ind
        if (self._cut_edges and gain < self._threshold) or (self._max_depth == node.depth) or (gain < 1e-6):
            node.type = self._major_type(data)
            return

        data_left = [None] * left_size
        data_right = [None] * right_size
        left_ind = 0
        right_ind = 0
        for x in data:
            if x[0][feature] <= key:
                data_left[left_ind] = x
                left_ind += 1
            else:
                data_right[right_ind] = x
                right_ind += 1
        node.left = Node()
        node.right = Node()
        node.left.depth = node.depth + 1
        node.right.depth = node.depth + 1
        node.key = key
        node.feature = feature
        self._build_tree(data_left, node.left)
        self._build_tree(data_right, node.right)

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
            self._cat_left = [0] * self._max_category
            self._cat_right = [0] * self._max_category
            for x in diff:
                self._cat_right[x] += 1
            for v in self._cat_right:
                ans -= self._gain_func(v / size)
        return ans

    def _major_type(self, data):
        cat = {}
        for x in data:
            if x[1] not in cat:
                cat[x[1]] = 1
            else:
                cat[x[1]] += 1
        ans = 0
        val = 0
        for k, v in cat.items():
            if v > val:
                val = v
                ans = k
        return ans
