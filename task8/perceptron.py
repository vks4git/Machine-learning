import numpy


class Perceptron:
    def __init__(self, input_size, epocs=10):
        self._learning_rate = 0.1
        self._layers = [numpy.zeros(input_size)]
        self._weights = []
        self._epocs = epocs

    def fit(self, X, Y):
        pass

    def add_layer(self, activation, weights=None):
        self._layers.append(numpy.array(activation))
        size = len(self._layers) - 1
        if weights is None:
            self._weights.append(numpy.random.uniform(-10, 10,
                                                      (len(self._weights[size - 1]), len(self._weights[size]))))
        else:
            self._weights.append(numpy.array(weights))
