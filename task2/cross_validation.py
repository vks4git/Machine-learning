import numpy
import pylab
import random
import math

__author__ = 'vks'


def k_fold(k, data_x, data_y, regression):
    error = 0.0
    for i in range(k):
        learning_x = []
        learning_y = []
        validation_x = []
        validation_y = []
        for j in range(len(data_y)):
            if j // k == i:
                validation_x.append(data_x[j])
                validation_y.append(data_y[j])
            else:
                learning_x.append(data_x[j])
                learning_y.append(data_y[j])
        regression.learn([learning_x], learning_y)
        error += regression.validate([validation_x], validation_y)
    return error / k


def leave_one_out(data_x, data_y, regression):
    error = 0.0
    for i in range(len(data_y)):
        learning_x = []
        learning_y = []
        validation_x = []
        validation_y = []
        for j in range(len(data_y)):
            if j == i:
                validation_x.append(data_x[j])
                validation_y.append(data_y[j])
            else:
                learning_x.append(data_x[j])
                learning_y.append(data_y[j])
        regression.learn([learning_x], learning_y)
        error += regression.validate([validation_x], validation_y)
    return error / len(data_y)


def monte_carlo(n, k, data_x, data_y, regression):
    error = 0.0
    for i in range(k):
        learning_x = []
        learning_y = []
        validation_x = []
        validation_y = []
        num = set()
        for j in range(n):
            p = 1
            while p in num:
                p = int(random.uniform(0, len(data_y) - 1))
            num.add(p)
        for j in range(len(data_y)):
            if j in num:
                validation_x.append(data_x[j])
                validation_y.append(data_y[j])
            else:
                learning_x.append(data_x[j])
                learning_y.append(data_y[j])
        regression.learn([learning_x], learning_y)
        error += regression.validate([validation_x], validation_y)
    return error / k


