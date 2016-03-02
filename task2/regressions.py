import numpy

__author__ = 'vks'


class LinearRegression:
    def __init__(self):
        self._vector = []

    def learn(self, x, y):
        A = numpy.matrix([[1 for i in range(len(y))]] + x).transpose()
        Y = numpy.matrix(y).transpose()
        self._vector = [beta for beta in (((A.transpose() * A) ** (-1)) * A.transpose() * Y).flat]
        self._vector.reverse()

    def get_coefficients(self):
        return self._vector

    def validate(self, x, y):
        error = 0.0
        for i in range(len(y)):
            predicted_y = 0.0
            for j in range(len(x)):
                predicted_y += x[j][i] * self._vector[j]
            predicted_y += self._vector[len(x)]
            error += (y[i] - predicted_y) ** 2
        return error


class FunctionalRegression:
    def __init__(self, functions):
        self._vector = []
        self._functions = functions

    def learn(self, x, y):
        matrix = [[f(xij) for xij in x[0]] for f in self._functions]
        lr = LinearRegression()
        lr.learn(matrix, y)
        self._vector = lr.get_coefficients()

    def get_coefficients(self):
        return self._vector

    def validate(self, x, y):
        error = 0.0
        for i in range(len(y)):
            predicted_y = 0.0
            for j in range(len(self._functions)):
                predicted_y += self._functions[j](x[0][i]) * self._vector[len(self._functions) - j - 1]
            predicted_y += self._vector[len(self._functions)]
            error += (y[i] - predicted_y) ** 2
        return error


class PolynomialRegression:
    def __init__(self, deg):
        self._vector = []
        self._fr = None
        self._deg = deg
        self._functions = [self._get_exponent(i) for i in range(1, self._deg + 1)]

    def learn(self, x, y):
        self._fr = FunctionalRegression(self._functions)
        self._fr.learn(x, y)
        self._vector = self._fr.get_coefficients()

    def get_coefficients(self):
        return self._fr.get_coefficients()

    def validate(self, x, y):
        return self._fr.validate(x, y)

    def _get_exponent(self, i):
        return lambda z: z ** i
