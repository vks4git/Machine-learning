from LocalMinimumSearch import gradient_descent_constant_lambda as gdcl, gradient_descent_decreasing_lambda as gddl, \
    gradient_descent_optimal_lambda as gdol, hybrid_gradient_descent as hgd, dichotomy_min as dm
import sys
from math import sin

__author__ = 'vks'


def f1(x):
    return x[0] ** 3 - 4 * x[0] ** 2 + 2 * x[0]


def grad_f1(x):
    return [3 * x[0] ** 2 - 8 * x[0] + 2]


def f2(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def grad_f2(x):
    return [-2 * (1 - x[0]) - 400 * (x[1] - x[0] ** 2) * x[0], 200 * (x[1] - x[0] ** 2)]


def f3(x):
    ans = 0
    for i in range(len(x) - 1):
        ans += (1 - x[i]) ** 2 + 100 * (x[i + 1] - x[i] ** 2) ** 2
    return ans


def grad_f3(x):
    ans = [-2 * (1 - x[0]) - 400 * (x[1] - x[0] ** 2) * x[0]]
    for i in range(1, len(x) - 1):
        ans += [-2 * (1 - x[i]) - 400 * (x[i + 1] - x[i] ** 2) * x[i] + 200 * (x[i] - x[i - 1] ** 2)]
    ans += [200 * (x[len(x) - 1] - x[len(x) - 2] ** 2)]
    return ans


def test_func1(x):
    return -5 ** (1 - (20 * x[0]) ** 2)


def grad_func1(x):
    return [5 ** (1 - (20 * x[0]) ** 2) * 64.3775164974 * x[0]]  # 64.3775164974 = 40 * ln5


def test_func2(x):
    return 0.1 * (sin(10 * x[0]) ** 2 * x[0] ** 2 + x[0] ** 2)


def grad_func2(x):
    return [0.1 * (10 * sin(20 * x[0]) * x[0] ** 2 + 2 * sin(10 * x[0]) ** 2 * x[0] + 2 * x[0])]


def test_func3(x):
    return ((x[0] ** 2 - 49) ** 2) / 20000


def grad_func3(x):
    return [4 * x[0] * (x[0] ** 2 - 49) / 20000]


def test_func4(x):
    return x[0] ** 2 * (sin(10 * x[0]) ** 2 + 1) + x[1] ** 2 * (sin(10 * x[1]) ** 2 + 1)


def grad_func4(x):
    return [10 * sin(20 * x[0]) * x[0] ** 2 + 2 * sin(10 * x[0]) ** 2 * x[0] + 2 * x[0],
            10 * sin(20 * x[1]) * x[1] ** 2 + 2 * sin(10 * x[1]) ** 2 * x[1] + 2 * x[1]]
