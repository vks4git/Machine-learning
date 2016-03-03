import pickle
from kNN import *
from metrics import *
from CV import *
from sklearn.neighbors import KNeighborsClassifier

__author__ = 'vks'


def grid_search(model, data_x, data_y):
    k = 1
    best_score = 0
    for i in range(1, 100):
        model.set_k(i)
        score = k_fold(10, data_x, data_y, model)
        if score > best_score:
            best_score = score
            k = i
    return k


with open("iris.txt", "rb") as f:
    data, types = pickle.load(f, encoding="latin-1")

knn = Naive_kNN(metrics=euclides)
k = grid_search(knn, data, types)
knn.set_k(k)
print("Optimal k: %s" % k)
print("CV result: %s" % k_fold(10, data, types, knn))
