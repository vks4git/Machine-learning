import idx2numpy
from skimage.feature import hog
from time import time
from sklearn import datasets
from numpy import array

px_x = 2
px_y = 2

iris = datasets.load_iris()
iris_x = []
iris_y = []
for i in range(150):
    if iris.target[i] == 0 or iris.target[i] == 1:
        iris_x.append(iris.data[i])
        iris_y.append(iris.target[i])

train_images = idx2numpy.convert_from_file("train-images.idx3-ubyte")
train_images_hog = array([hog(img, orientations=8, pixels_per_cell=(px_x, px_y), cells_per_block=(1, 1))
                          for img in train_images])
train_labels = idx2numpy.convert_from_file("train-labels.idx1-ubyte")
test_images = idx2numpy.convert_from_file("t10k-images.idx3-ubyte")
test_images_hog = array([hog(img, orientations=8, pixels_per_cell=(px_x, px_y), cells_per_block=(1, 1))
                         for img in test_images])
test_labels = idx2numpy.convert_from_file("t10k-labels.idx1-ubyte")


def shift(axis, dist, data):
    ans = [[[0] * len(data[0][0]) for i in range(len(data[0]))] for j in range(len(data))]
    for i in range(len(data)):
        size = len(ans[i])
        for j in range(size):
            for k in range(size):
                if axis == "y":
                    ans[i][k][j] = data[i][(k + dist) % size][j]
                elif axis == "x":
                    ans[i][j][k] = data[i][j][(k + dist) % size]
    return ans


def rotate(data):
    ans = [[[0] * len(data[0][0]) for i in range(len(data[0]))] for j in range(len(data))]
    for i in range(len(data)):
        size = len(ans[i])
        for j in range(size):
            for k in range(size):
                ans[i][j][k] = data[i][size - k - 1][j]
    return ans


def flip(axis, data):
    ans = [[[0] * len(data[0][0]) for i in range(len(data[0]))] for j in range(len(data))]
    for i in range(len(data)):
        size = len(ans[i])
        for j in range(size):
            for k in range(size):
                if axis == "x":
                    ans[i][j][k] = data[i][size - j - 1][k]
                elif axis == "y":
                    ans[i][j][k] = data[i][j][size - k - 1]
    return ans


def validate(model, data, name):
    start = time()
    data = data
    errors = 0
    for i in range(len(data)):
        if model.predict(data[i]) != test_labels[i]:
            errors += 1

    finish = time()
    print("*** Validating %s took %s s. Errors: %s (%s%%)\n" %
          (name, finish - start, errors, errors / 100))


print("Loading completed. Transforming images...")

img_shift_x0y1 = array(shift("y", 1, test_images))
img_shift_x0y1_hog = array([hog(img, orientations=8, pixels_per_cell=(px_x, px_y), cells_per_block=(1, 1))
                            for img in img_shift_x0y1])

img_shift_x0y5 = array(shift("y", 5, test_images))
img_shift_x0y5_hog = array([hog(img, orientations=8, pixels_per_cell=(px_x, px_y), cells_per_block=(1, 1))
                            for img in img_shift_x0y5])

img_shift_x0y10 = array(shift("y", 10, test_images))
img_shift_x0y10_hog = array([hog(img, orientations=8, pixels_per_cell=(px_x, px_y), cells_per_block=(1, 1))
                             for img in img_shift_x0y10])

img_shift_x1y0 = array(shift("x", 1, test_images))
img_shift_x1y0_hog = array([hog(img, orientations=8, pixels_per_cell=(px_x, px_y), cells_per_block=(1, 1))
                            for img in img_shift_x1y0])

img_shift_x5y0 = array(shift("x", 5, test_images))
img_shift_x5y0_hog = array([hog(img, orientations=8, pixels_per_cell=(px_x, px_y), cells_per_block=(1, 1))
                            for img in img_shift_x5y0])

img_shift_x10y0 = array(shift("x", 10, test_images))
img_shift_x10y0_hog = array([hog(img, orientations=8, pixels_per_cell=(px_x, px_y), cells_per_block=(1, 1))
                             for img in img_shift_x10y0])

img_rotate = array(rotate(test_images))
img_rotate_hog = array([hog(img, orientations=8, pixels_per_cell=(px_x, px_y), cells_per_block=(1, 1))
                        for img in img_rotate])

img_flip_x = array(flip("x", test_images))
img_flip_x_hog = array([hog(img, orientations=8, pixels_per_cell=(px_x, px_y), cells_per_block=(1, 1))
                        for img in img_flip_x])

img_flip_y = array(flip("y", test_images))
img_flip_y_hog = array([hog(img, orientations=8, pixels_per_cell=(px_x, px_y), cells_per_block=(1, 1))
                        for img in img_flip_y])

print("Transformation completed.")
