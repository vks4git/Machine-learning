import math
import random

__author__ = 'vks'

eps = 1e-9
inf = 2 ** 256


def vec_sub(x, y):
    return [x[i] - y[i] for i in range(len(x))]


def vec_add(x, y):
    return [x[i] + y[i] for i in range(len(x))]


def vec_norm(x):
    ans = 0
    for i in x:
        ans += i ** 2
    return math.sqrt(ans)


def vec_mul(a, x):
    return [i * a for i in x]


def dichotomy_min(func, a, b):
    sufficient_precision = False
    x0 = []
    while not sufficient_precision:
        sufficient_precision = True
        x0 = [(a[i] + b[i]) / 2 for i in range(len(a))]
        delta = [(b[i] - x0[i]) / 2 for i in range(len(a))]
        for i in range(len(a)):
            left = [j for j in x0]
            right = [j for j in x0]
            left[i] = x0[i] - delta[i]
            right[i] = x0[i] + delta[i]
            if func(left) < func(right):
                b[i] = x0[i] + delta[i]
            else:
                a[i] = x0[i] - delta[i]
        for i in range(len(a)):
            if abs(b[i] - a[i]) >= eps:
                sufficient_precision = False
    return x0, func(x0)


def gradient_descent_optimal_lambda(func, grad, a, b, x0, l):
    xj = [i for i in x0]
    x_prev = vec_sub(x0, vec_mul(l, grad(x0)))
    while abs(func(xj) - func(x_prev)) >= eps:
        area = 0
        for i in range(len(a)):
            area = max(area, abs(x_prev[i] - xj[i]))
        area *= 2
        min_f = inf
        for i in range(10000):
            f = func(vec_sub(xj, vec_mul(area * i / 10000, grad(xj))))
            if f < min_f:
                l = area * i / 10000
                min_f = f
        x_prev = [i for i in xj]
        xj = vec_sub(xj, vec_mul(l, grad(xj)))
        for i in range(len(xj)):
            xj[i] = max(xj[i], a[i])
            xj[i] = min(xj[i], b[i])
    return xj, func(xj)


def gradient_descent_constant_lambda(func, grad, a, b, x0, l):
    xj = [i for i in x0]
    for j in range(100000):
        xj = vec_sub(xj, vec_mul(l, grad(xj)))
        for i in range(len(xj)):
            xj[i] = max(xj[i], a[i])
            xj[i] = min(xj[i], b[i])
    return xj, func(xj)


def gradient_descent_decreasing_lambda(func, grad, a, b, x0, l):
    xj = [i for i in x0]
    x_prev = vec_sub(x0, vec_mul(l, grad(x0)))
    while abs(func(xj) - func(x_prev)) >= eps:
        x_prev = [i for i in xj]
        l *= 0.95
        xj = vec_sub(xj, vec_mul(l, grad(xj)))
        for i in range(len(xj)):
            xj[i] = max(xj[i], a[i])
            xj[i] = min(xj[i], b[i])
    return xj, func(xj)


def hybrid_gradient_descent(func, grad, a, b, l):
    min_x = [inf for i in range(len(a))]
    min_f = inf
    for i in range(1000000):
        x0 = [random.uniform(a[i], b[i]) for i in range(len(a))]
        f = func(x0)
        if f < min_f:
            min_f = f
            min_x = x0
    return gradient_descent_optimal_lambda(func, grad, a, b, min_x, l)
