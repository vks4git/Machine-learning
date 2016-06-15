from tensorflow.examples.tutorials.mnist import input_data
from main_t10 import evaluate
from scipy.optimize import fmin_l_bfgs_b

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

result, f, dict = fmin_l_bfgs_b(lambda x: -evaluate(x, mnist), [1000, 100, 0.5, 15, 250], maxiter=1000, approx_grad=True,
                                bounds=[(10, 10000), (10, 1000), (0.1, 0.9), (5, 5000), (5, 5000)], maxfun=1000)
print(result, f)
