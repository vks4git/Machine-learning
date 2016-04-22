import numpy
from random import uniform
from math import log


class K_means:
    def __init__(self, k, maxiter=2 ** 256):
        self._clusters = k
        self._distribution = None
        self._iter = maxiter

    def fit(self, data):
        area = numpy.array([(0.0, 0.0)] * len(data[0]))
        clusters = numpy.array([0] * len(data))
        for x in data:
            for i in range(len(area)):
                area[i][0] = min(area[i][0], x[i])
                area[i][1] = max(area[i][1], x[i])
        # choose random k points
        centers = numpy.array([[uniform(area[i][0], area[i][1]) for i in range(len(data[0]))]
                               for j in range(self._clusters)])
        converges = False
        it = 0
        while not converges or it < self._iter:
            converges = True
            it += 1
            for i in range(len(data)):
                dist = 2 ** 256
                cluster = -1
                for j in range(len(centers)):
                    current = self._euclidean(data[i], centers[j])
                    if current < dist:
                        cluster = j
                if clusters[i] != cluster:
                    converges = False
                clusters[i] = cluster
            for i in range(self._clusters):
                centers[i] = numpy.multiply(centers[i], 0)
            count = numpy.zeros(self._clusters)
            for i in range(len(data)):
                centers[clusters[i]] = numpy.sum([centers[clusters[i]], data[i]], axis=0)
                count[clusters[i]] += 1
            for i in range(self._clusters):
                centers[i] = numpy.multiply(centers[i], 1 / count[i] if count[i] != 0 else 1)
        self._distribution = clusters

    def score(self, labels):
        sizes = numpy.zeros(self._clusters)
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

    def _euclidean(self, x, y):
        r = numpy.sum([x, numpy.multiply(y, -1)], axis=0)
        return numpy.dot(r, r)
