from svm import SVM_Linear
from svm import SVM_Nonlinear
from random import uniform
from numpy import dot

x = [[uniform(5, 10), uniform(3, 6)] for i in range(10)]
x += [[uniform(-5, -10), uniform(-3, -6)] for i in range(10)]
y = [1] * 10 + [-1] * 10
svm = SVM_Linear()
svm.fit(x, y)
print(svm.predict([7, 4]))
