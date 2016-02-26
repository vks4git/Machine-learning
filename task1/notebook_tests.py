from LocalMinimumSearch import gradient_descent_constant_lambda as gdcl, gradient_descent_decreasing_lambda as gddl, \
    gradient_descent_optimal_lambda as gdol, hybrid_gradient_descent as hgd, dichotomy_min as dm

from Research import *
import sys
import random

__author__ = 'vks'


def test_func(x):
    return test_func4(x)


def grad_func(x):
    return grad_func4(x)


def prefix(i):
    if i == 1:
        return "1st "
    if i == 2:
        return "2nd "
    if i == 3:
        return "3rd "
    return str(i) + "th "


for i in range(1, 11):
    sys.stdout.write(prefix(i) + "iteration: " + str(
        gdcl(test_func, grad_func, [-10, -10], [10, 10], [random.uniform(-10, 10), random.uniform(-10, 10)],
             i / 10)) + "\n" * 2)

sys.stdout.write("\n" * 3)

for i in range(1, 11):
    sys.stdout.write(prefix(i) + "iteration: " + str(
        gddl(test_func, grad_func, [-10, -10], [10, 10], [random.uniform(-10, 10), random.uniform(-10, 10)],
             i / 10)) + "\n" * 2)

sys.stdout.write("\n" * 3)

for i in range(1, 11):
    sys.stdout.write(prefix(i) + "iteration: " + str(
        gdol(test_func, grad_func, [-10, -10], [10, 10], [random.uniform(-10, 10), random.uniform(-10, 10)],
             i / 10)) + "\n" * 2)

sys.stdout.write("\n" * 3)

for i in range(1, 11):
    sys.stdout.write(
        prefix(i) + "iteration: " + str(hgd(test_func, grad_func, [-10, -10], [10, 10], i / 10)) + "\n" * 2)
