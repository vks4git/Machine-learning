from random_forest import RandomForest
from CV import k_fold
import pickle

__author__ = 'vks'

with open("iris.txt", "rb") as f:
    data, types = pickle.load(f, encoding="latin-1")

forest = RandomForest(size=200, features=0.2)
print(k_fold(10, data, types, forest))

