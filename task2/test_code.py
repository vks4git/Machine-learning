from regressions import *
import math
import numpy
import pylab
import random

__author__ = 'vks'

sample_length = 100

data_x = [[0.1 * i for i in range(sample_length)]]
data_linear = [3.4 * data_x[0][i] + 6 + random.uniform(-3, 3) for i in range(sample_length)]
data_cubic = [data_x[0][i] * (data_x[0][i] - 15) * (data_x[0][i] - 30) / 125 + 5 +
              random.uniform(-3, 3) for i in range(sample_length)]
data_functional = [15 * math.cos(data_x[0][i]) + 0.1 * (2 ** data_x[0][i]) - 0.03 * (data_x[0][i] ** 3) +
                   random.uniform(-3, 3) for i in range(sample_length)]

lr = LinearRegression()
lr.learn(data_x, data_linear)
vec = lr.get_coefficients()
data_linear_regression = [data_x[0][i] * vec[0] + vec[1] for i in range(sample_length)]

pr = PolynomialRegression(3)
pr.learn(data_x, data_cubic)
vec = pr.get_coefficients()
data_polynomial_regression = [vec[0] * data_x[0][i] ** 3 + vec[1] * data_x[0][i] ** 2 +
                              vec[2] * data_x[0][i] + vec[3] for i in range(sample_length)]

fr = FunctionalRegression([lambda x: math.cos(x), lambda x: 2 ** x, lambda x: x ** 3])
fr.learn(data_x, data_functional)
vec = fr.get_coefficients()
data_functional_regression = [vec[2] * math.cos(data_x[0][i]) + vec[1] * (2 ** data_x[0][i]) +
                              vec[0] * data_x[0][i] ** 3 + vec[3] for i in range(sample_length)]

pylab.xlabel("x")
pylab.ylabel("y")
pylab.plot(data_x[0], data_functional, 'ro')
pylab.plot(data_x[0], data_functional_regression, label="functional regression", linewidth=3)
pylab.plot(data_x[0], data_cubic, 'm^')
pylab.plot(data_x[0], data_polynomial_regression, label="polynomial regression", linewidth=3)
pylab.plot(data_x[0], data_linear, 'cs')
pylab.plot(data_x[0], data_linear_regression, label="linear regression", linewidth=3)
pylab.legend(loc='upper left', title="Regressions")
pylab.show()
