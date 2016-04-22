import numpy
from scipy.spatial.distance import minkowski
from math import log


class DBScan:
    def __init__(self, rad, min_pts):
        self._rad = rad
        self._min_pts = min_pts
        self._data = None
        self._clusters = 0
        self._distribution = None
        self._size = 0

    def fit(self, data):
        self._data = data
        size = len(data)
        self._size = size
        clusters = numpy.zeros(size, dtype="int32")
        visited = numpy.zeros(size, dtype="int32")
        current_cluster = 1
        for i in range(size):
            if visited[i] == 1:
                continue
            visited[i] = 1
            region = self._neighbours(i)
            if len(region) < self._min_pts:
                continue
            self._expand(i, current_cluster, clusters, region, visited)
            current_cluster += 1
        self._clusters = current_cluster
        self._distribution = clusters

    def score(self, labels):
        sizes = numpy.zeros(self._clusters + 1, dtype="int32")
        classes = numpy.array([{} for i in range(self._clusters)])
        for i in range(len(self._distribution)):
            sizes[self._distribution[i]] += 1
            if labels[i] not in classes[self._distribution[i]]:
                classes[self._distribution[i]][labels[i]] = 1
            else:
                classes[self._distribution[i]][labels[i]] += 1
        entropys = numpy.zeros(self._clusters)
        for i in range(self._clusters):
            for v in classes[i].values():
                entropys[i] -= v / sizes[i] * log(v / sizes[i], 2)
        return numpy.sum(entropys) / self._clusters

    def clusters(self):
        return self._clusters

    def _neighbours(self, ind):
        ans = []
        for i in range(self._size):
            dist = minkowski(self._data[i], self._data[ind], 3)
            if dist <= self._rad:
                ans.append(i)
        return numpy.array(ans)

    def _expand(self, ind, c, clusters, region, visited):
        clusters[ind] = c
        for i in region:
            if visited[i] == 0:
                visited[i] = 1
                new_reg = self._neighbours(i)
                if len(new_reg) >= self._min_pts:
                    region = numpy.append(region, new_reg)
            if clusters[i] == 0:
                clusters[i] = c
