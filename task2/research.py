import pickle
from cross_validation import *
from regressions import *

__author__ = 'vks'

x, y = pickle.load(open("/home/vks/Desktop/MachineLearning/task2/data/task2_dataset_1.txt", "rb"), encoding="latin-1")
fr = FunctionalRegression([lambda z: math.cos(z), lambda z: math.cos(2 * z), lambda z: math.cos(3 * z)])
x = [[t for t in numpy.matrix(x).transpose().flat]]
y = [t for t in numpy.matrix(y).transpose().flat]
fr.learn(x, y)
vec = fr.get_coefficients()
reg_y = [vec[2] * math.cos(t) + vec[1] * math.cos(2 * t) + vec[0] * math.cos(3 * t) + vec[3] for t in sorted(x[0])]

pylab.xlabel("x")
pylab.ylabel("y")
pylab.plot(x[0], y, "ro")
pylab.plot(sorted(x[0]), reg_y, linewidth=3)
pylab.show()

print("K-fold CV: %s" % k_fold(10, x[0], y, fr))
print("Leave-one-out CV: %s" % leave_one_out(x[0], y, fr))
print("Monte Carlo CV: %s" % monte_carlo(10, 10, x[0], y, fr))
