from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

batch_size = 128
nb_epoch = 10
layer1 = 300
layer2 = 300


def evaluate(lr, pos):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = (X_train.astype('float32')).reshape((60000, 784))
    X_test = (X_test.astype('float32')).reshape((10000, 784))
    X_train /= 255
    X_test /= 255

    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    model = Sequential()
    model.add(Dense(output_dim=layer1, input_dim=784))
    if pos == 0:
        model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dense(output_dim=layer2, input_dim=layer1))
    if pos == 1:
        model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dense(output_dim=10, input_dim=layer2))
    if pos == 2:
        model.add(BatchNormalization())
    model.add(Activation("softmax"))

    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=lr, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=0, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    return score[1]


for i in range(1, 10):
    for j in range(3):
        print("Accuracy: %s, learning rate: %s, position: after layer %i" % (evaluate(i / 100, 0), i / 100, j))
