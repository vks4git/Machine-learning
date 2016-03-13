from random_forest import RandomForest
import pickle
from import_labs import import_labs
from time import time

__author__ = 'vks'

import_labs(["task3/"])
from CV import k_fold
from kNN import Naive_kNN

with open("iris.txt", "rb") as f:
    data, types = pickle.load(f, encoding="latin-1")

forest = RandomForest(size=10, features=0.5)
print(k_fold(10, data, types, forest))
knn = Naive_kNN()
print(k_fold(10, data, types, knn))
