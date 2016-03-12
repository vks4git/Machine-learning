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
        self._root = None
        self._threshold = threshold
        self._gain_func = gain_func

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

    def _gini(self, categories):
        cat = {}
        for x in categories:
            if x not in cat:
                cat[x] = 1
            else:
                cat[x] += 1
        ans = 0
        for v in cat.values():
            ans += (v / len(categories)) ** 2
        return 1 - ans

    def _entropy(self, categories):
        cat = {}
        for x in categories:
            if x not in cat:
                cat[x] = 1
            else:
                cat[x] += 1
        ans = 0
        for v in cat.values():
            ans -= v / len(categories) * math.log(v / len(categories), 2)
        return ans

    def _gain(self, data, categories):
        feature = -1
        key = -1
        gain = 0
        dataset = [(data[i], categories[i]) for i in range(len(categories))]
        for i in range(len(data[0])):
            dataset.sort(key=lambda x: x[0][i])
            cat_left = []
            cat_right = [j[1] for j in dataset]
            ind = 0
            while len(cat_right) > 0:
                cat_left.append(cat_right.pop())
                ind += 1
                while len(cat_right) > 0 and dataset[ind][0][i] == dataset[ind - 1][0][i]:
                    cat_left.append(cat_right.pop())
                    ind += 1
                test_gain = 0.0
                if self._gain_func == 'entropy':
                    test_gain = self._entropy(categories) - \
                                len(cat_left) / len(categories) * self._entropy(cat_left) - \
                                len(cat_right) / len(categories) * self._entropy(cat_right)
                elif self._gain_func == 'gini':
                    test_gain = self._gini(categories) - \
                                len(cat_left) / len(categories) * self._gini(cat_left) - \
                                len(cat_right) / len(categories) * self._gini(cat_right)
                if test_gain > gain:
                    gain = test_gain
                    feature = i
                    key = dataset[ind - 1][0][i]
        return feature, key, gain
