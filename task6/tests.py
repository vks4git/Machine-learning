from idx2numpy import convert_from_file
from adaboost import AdaBoostMulticlass
from numpy import reshape, multiply

train_images = reshape(convert_from_file("train-images.idx3-ubyte").astype('float64'), (60000, 784))
train_labels = convert_from_file("train-labels.idx1-ubyte")
test_images = reshape(convert_from_file("t10k-images.idx3-ubyte").astype('float64'), (10000, 784))
test_labels = convert_from_file("t10k-labels.idx1-ubyte")

train_images = multiply(train_images, 1 / 255)
test_images = multiply(test_images, 1 / 255)

ada = AdaBoostMulticlass(1000)
ada.fit(train_images, train_labels)
score = 0
for i in range(10000):
    if ada.predict(test_images[i]) == test_labels[i]:
        score += 1

print(score / 10000)
