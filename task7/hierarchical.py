import csv
import numpy
from QuickWeightedUnionFind import QuickWeightedUF
from scipy.spatial.distance import euclidean


class Hierarchical:
    def __init__(self, rad):
        self._data = None
        self._rad = rad

    def load_from_csv(self, path):
        self._data = []
        maxlst = numpy.zeros(8)
        with open(path) as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] != "PassengerId":
                    lst = self._list2vec(row)
                    for i in range(len(lst)):
                        maxlst[i] = max(maxlst[i], lst[i])
                    self._data.append(lst)

        self._data = numpy.array(self._data)
        for i in range(len(self._data)):
            for j in range(len(self._data[i])):
                self._data[i][j] /= maxlst[j]
        f.close()
        uf = QuickWeightedUF(len(self._data))
        finished = False
        size = len(self._data)
        while not finished:
            finished = True
            for i in range(size):
                for j in range(i + 1, size):
                    if euclidean(self._data[i], self._data[j]) <= self._rad and not uf.same_set(i, j):
                        finished = False
                        uf.union(i, j)

    def _list2vec(self, list):
        ans = []
        for i in range(len(list)):
            if i == 0 or i == 3 or i == 8 or i == 10:
                continue
            if i == 1 or i == 2 or i == 6 or i == 7:
                if list[i] == "":
                    ans.append(0)
                else:
                    ans.append(int(list[i]))
            if i == 4:
                if list[i] == "":
                    ans.append(2)
                else:
                    if list[i] == "male":
                        ans.append(0)
                    else:
                        ans.append(1)
            if i == 5 or i == 9:
                if list[i] == "":
                    ans.append(0)
                else:
                    ans.append(float(list[i]))
            if i == 11:
                if list[i] == "":
                    ans.append(4)
                elif list[i] == "C":
                    ans.append(0)
                elif list[i] == "S":
                    ans.append(1)
                else:
                    ans.append(2)
        return numpy.array(ans, dtype="float32")
