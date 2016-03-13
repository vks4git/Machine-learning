import math

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
    def __init__(self, maxdepth=-1, cut_edges=False, gain_func='gini', threshold=1e-2):
        self._maxdepth = maxdepth
        self._cut_edges = cut_edges
        self._data = []
        self._categories = []
        self._cat_right = {}
        self._cat_left = {}
        self._root = None
        self._threshold = threshold
        if gain_func == 'gini':
            self._gain_func = self._gini
        elif gain_func == 'entropy':
            self._gain_func = self._entropy

    def fit(self, data, categories):
        self._data = data
        self._categories = categories
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

    def set_maxdepth(self, val):
        self._maxdepth = val

    def perform_edges_cut(self, val):
        self._cut_edges = val

    def _build_tree(self, data, categories, node):
        gain = 0
        feature = 0
        key = 0
        if len(data) > 1:
            feature, key, gain = self._gain(data, categories)
        if (self._cut_edges and gain < self._threshold) or (self._maxdepth == node.depth) or (gain < 1e-6):
            cat = {}
            for x in categories:
                if x not in cat:
                    cat[x] = 1
                else:
                    cat[x] += 1
            type = 0
            val = 0
            for k, v in cat.items():
                if v > val:
                    val = v
                    type = k
            node.type = type
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

    def _gini(self, mode, diff, size):
        if size == 0:
            return 1
        ans = 0
        if mode == 'right':
            for x in diff:
                self._cat_right[x] -= 1
            for v in self._cat_right.values():
                ans += (v / size) ** 2
        elif mode == 'left':
            for x in diff:
                if x not in self._cat_left:
                    self._cat_left[x] = 1
                else:
                    self._cat_left[x] += 1
            for v in self._cat_left.values():
                ans += (v / size) ** 2
        elif mode == 'init':
            self._cat_right.clear()
            self._cat_left.clear()
            for x in diff:
                if x not in self._cat_right:
                    self._cat_right[x] = 1
                else:
                    self._cat_right[x] += 1
            for v in self._cat_right.values():
                ans += (v / size) ** 2
        return 1 - ans

    def _entropy(self, mode, diff, size):
        if size == 0:
            return 1
        ans = 0
        if mode == 'right':
            for x in diff:
                self._cat_right[x] -= 1
            for v in self._cat_right.values():
                ans -= v / size * math.log(v / size, 2)
        elif mode == 'left':
            for x in diff:
                if x not in self._cat_left:
                    self._cat_left[x] = 1
                else:
                    self._cat_left[x] += 1
            for v in self._cat_left.values():
                ans -= v / size * math.log(v / size, 2)
        elif mode == 'init':
            self._cat_right.clear()
            self._cat_left.clear()
            for x in diff:
                if x not in self._cat_right:
                    self._cat_right[x] = 1
                else:
                    self._cat_right[x] += 1
            for v in self._cat_right.values():
                ans -= v / size * math.log(v / size, 2)
        return ans

    def _gain(self, data, categories):
        feature = -1
        key = -1
        gain = 0
        dataset = [(data[i], categories[i]) for i in range(len(categories))]
        size = len(categories)
        for i in range(len(data[0])):
            dataset.sort(key=lambda x: x[0][i])
            cat_right = [j[1] for j in dataset]
            overall_gain = self._gain_func('init', cat_right, size)
            ind = 0
            while len(cat_right) > 0:
                diff = [cat_right.pop()]
                ind += 1
                while len(cat_right) > 0 and dataset[ind][0][i] == dataset[ind - 1][0][i]:
                    diff.append(cat_right.pop())
                    ind += 1
                test_gain = overall_gain - \
                            ind / size * self._gain_func('left', diff, ind) - \
                            len(cat_right) / size * self._gain_func('right', diff, len(cat_right))
                if test_gain > gain:
                    gain = test_gain
                    feature = i
                    key = dataset[ind - 1][0][i]
        return feature, key, gain
