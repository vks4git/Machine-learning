import numpy

__author__ = 'vks'


class QuickWeightedUF:
    def __init__(self, size):
        self._parents = numpy.array([i for i in range(size)])
        self._weights = numpy.ones(size)

    def same_set(self, u, v):
        return self._root(u) == self._root(v)

    def union(self, u, v):
        if self.same_set(u, v):
            return
        root_u = self._root(u)
        root_v = self._root(v)
        if self._weights[root_u] > self._weights[root_v]:
            self._parents[root_v] = root_u
            self._weights[root_u] += self._weights[root_v]
        else:
            self._parents[root_u] = root_v
            self._weights[root_v] += self._weights[root_u]

    def _root(self, v):
        if v == self._parents[v]:
            return v
        self._parents[v] = self._root(self._parents[v])
        return self._parents[v]