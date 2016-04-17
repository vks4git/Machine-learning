from scipy.optimize import minimize
from numpy import multiply
from numpy import sum
from numpy import dot


class SVM_Linear:
    def __init__(self):
        self._data_x = []
        self._data_y = []
        self._w = []
        self._w0 = 0

    def fit(self, data_x, data_y):
        self._data_x = data_x
        self._data_y = data_y
        x0 = [0] * len(data_y)
        cons = {
            "type": "eq",
            "fun": lambda z: dot(z, self._data_y),
        }
        lambdas = minimize(lambda z: self._sum(z, data_x, data_y), x0, method="SLSQP",
                           jac=lambda z: self._jac(z, data_x, data_y), constraints=cons,
                           bounds=([(0, None)] * len(data_y))).x
        self._w = [0] * len(data_x[0])
        for i in range(len(data_y)):
            self._w = sum([self._w, multiply(data_x[i], data_y[i] * lambdas[i])], axis=0)
        self._w0 = dot(data_x[0], self._w) - data_y[0]

    def _sum(self, z, x, y):
        ans = 0
        for i in range(len(y)):
            for j in range(len(y)):
                ans += z[i] * z[j] * y[i] * y[j] * dot(x[i], x[j]) / 2
        for i in z:
            ans -= i
        return ans

    def _jac(self, z, x, y):
        ans = []
        for i in range(len(y)):
            ans.append(0)
            for j in range(len(y)):
                if i != j:
                    ans[i] += z[j] * y[i] * y[j] * dot(x[i], x[j]) / 2
                else:
                    ans[i] += z[j] * y[i] * y[j] * dot(x[i], x[j])
            ans[i] -= self._sign(z[i])
        return ans

    def _sign(self, n):
        if n < 0:
            return -1
        return 1

    def predict(self, x):
        return self._sign(dot(self._w, x) - self._w0)


class SVM_Nonlinear:
    def __init__(self, kernel_func, c):
        self._data_x = []
        self._data_y = []
        self._w = []
        self._w0 = 0
        self._kernel_func = kernel_func
        self._c = c

    def fit(self, data_x, data_y):
        self._data_x = data_x
        self._data_y = data_y
        x0 = [self._c / 2] * len(data_y)
        cons = [{
            "type": "eq",
            "fun": lambda z: dot(z, self._data_y),
            "jac": lambda z: data_y
        }]
        lambdas = minimize(lambda z: self._sum(z, data_x, data_y), x0, method="SLSQP",
                           jac=lambda z: self._jac(z, data_x, data_y), constraints=cons,
                           bounds=([(0, self._c)] * len(data_y))).x
        self._w = [0] * len(data_x[0])
        for i in range(len(data_y)):
            self._w = sum([self._w, multiply(data_x[i], data_y[i] * lambdas[i])], axis=0)
        self._w0 = self._kernel_func(data_x[0], self._w) - data_y[0]

    def _sum(self, z, x, y):
        ans = 0
        for i in range(len(y)):
            for j in range(len(y)):
                ans += z[i] * z[j] * y[i] * y[j] * self._kernel_func(x[i], x[j]) / 2
        for i in z:
            ans -= i
        return ans

    def _jac(self, z, x, y):
        ans = []
        for i in range(len(y)):
            ans.append(0)
            for j in range(len(y)):
                if i != j:
                    ans[i] += z[j] * y[i] * y[j] * dot(x[i], x[j]) / 2
                else:
                    ans[i] += z[j] * y[i] * y[j] * dot(x[i], x[j])
            ans[i] -= self._sign(z[i])
        return ans

    def _sign(self, n):
        if n < 0:
            return -1
        return 1

    def predict(self, x):
        return self._sign(dot(self._w, x) - self._w0)


class SVM_Linear_Multiclass:
    def __init__(self):
        self._classificators = []
        self._count = 0

    def fit(self, data_x, data_y):
        self._count = len(data_y)
        for i in range(self._count):
            self._classificators.append(SVM_Linear())
            new_y = []
            for j in range(self._count):
                if data_y[i] == j:
                    new_y.append(1)
                else:
                    new_y.append(-1)

            self._classificators[i].fit(data_x, new_y)

    def predict(self, x):
        votes = [0] * self._count
        for i in range(self._count):
            votes[i] += self._classificators[i].predict(x)
        ans = -2 ** 256
        for i in votes:
            ans = max(ans, i)
        for i in range(len(votes)):
            if votes[i] == ans:
                return i


class SVM_Nonlinear_Multiclass:
    def __init__(self, kernel_func, c):
        self._classificators = []
        self._count = 0
        self._kernel_func = kernel_func
        self._c = c

    def fit(self, data_x, data_y):
        self._count = len(data_y)
        for i in range(self._count):
            self._classificators.append(SVM_Nonlinear(self._kernel_func, self._c))
            new_y = []
            for j in range(self._count):
                if data_y[i] == j:
                    new_y.append(1)
                else:
                    new_y.append(-1)

            self._classificators[i].fit(data_x, new_y)

    def predict(self, x):
        votes = [0] * self._count
        for i in range(self._count):
            votes[i] += self._classificators[i].predict(x)
        ans = -2 ** 256
        for i in votes:
            ans = max(ans, i)
        for i in range(len(votes)):
            if votes[i] == ans:
                return i
