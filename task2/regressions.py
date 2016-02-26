import numpy
import pylab
import random

__author__ = 'vks'

sample_length = 300


def get_exponent(i):
    return lambda z: z ** i


def linear_regression(x, y):
    A = numpy.matrix([[1 for i in range(sample_length)]] + x).transpose()
    y = numpy.matrix(y).transpose()
    return ((A.transpose() * A) ** (-1)) * A.transpose() * y


def functional_regression(x, y, functions):
    matrix = [[f(xij) for xij in x[0]] for f in functions]
    return linear_regression(matrix, y)


def polynomial_regression(x, y, n):
    return functional_regression(x, y, [get_exponent(i) for i in range(1, n + 1)])


data_x = [[0.1 * i for i in range(sample_length)]]
data_linear = [3.4 * data_x[0][i] + 6 + random.uniform(-3, 3) for i in range(sample_length)]
data_cubic = [data_x[0][i] * (data_x[0][i] - 15) * (data_x[0][i] - 30) / 125 + 5 +
              random.uniform(-3, 3) for i in range(sample_length)]

vec = linear_regression(data_x, data_linear)  # should be close to 3.4 and 6
data_linear_regression = [data_x[0][i] * vec.flat[1] + vec.flat[0] for i in range(sample_length)]

vec = polynomial_regression(data_x, data_cubic, 3)  # should be close to 1/125, -9/25, 18/5 and 5
data_polynomial_regression = [vec.flat[3] * data_x[0][i] ** 3 + vec.flat[2] * data_x[0][i] ** 2 +
                              vec.flat[1] * data_x[0][i] + vec.flat[0] for i in range(sample_length)]

pylab.xlabel("x")
pylab.ylabel("y")
pylab.plot(data_x[0], data_linear, 'ro')
pylab.plot(data_x[0], data_linear_regression, label="linear regression", linewidth=3)
pylab.plot(data_x[0], data_cubic, 'm^')
pylab.plot(data_x[0], data_polynomial_regression, label="polynomial regression", linewidth=3)
pylab.legend(loc='upper left', title="Regressions")
pylab.show()

pylab.xlabel("x")
pylab.ylabel("y")
pylab.plot([i / 1000 for i in range(-100, 100)], [i ** 2 for i in range(-100, 100)])
pylab.show()
