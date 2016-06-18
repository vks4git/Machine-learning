from keras.datasets import mnist
import numpy
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from scipy.special import expit
import os


def evaluate(generating_train_percentage, nb_epoch, dim, wanted_digit, save_model=False):
    (X_train, y_train), _ = mnist.load_data()

    X_train = numpy.reshape(X_train, (X_train.shape[0], numpy.multiply(X_train.shape[1], X_train.shape[2])))
    X_train = X_train.astype('float32')
    X_train /= float(255)
    wanted_digits = []
    for i in range(X_train.shape[0]):
        if y_train[i] == wanted_digit:
            wanted_digits.append(X_train[i])
    wanted_digits = numpy.array(wanted_digits)

    desc = Sequential()
    gen = Sequential()

    desc.add(Dense(input_dim=784, output_dim=250))
    desc.add(Activation('sigmoid'))
    desc.add(Dense(1))
    desc.add(Activation('sigmoid'))

    gen.add(Dense(input_dim=3000, output_dim=1500))
    gen.add(Activation('relu'))
    gen.add(Dense(784))
    gen.add(Activation('sigmoid'))

    desc.trainable = False
    gen.add(desc)
    gen.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True),
                metrics=['accuracy'])

    desc.trainable = True
    desc.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True),
                 metrics=['accuracy'])

    batch_size = 32
    fig = plt.figure()
    fixed_noise = numpy.random.rand(1, dim).astype('float32')
    generating_train_percentage = int(1 / generating_train_percentage)
    if not os.path.exists(str(wanted_digit) + "/"):
        os.makedirs(str(wanted_digit))
    for iter in range(nb_epoch):
        gen_acc = 0
        desc_acc = 0
        gen_count = 0
        desc_count = 0
        for (first, last) in zip(range(0, wanted_digits.shape[0] - batch_size, batch_size),
                                 range(batch_size, wanted_digits.shape[0], batch_size)):
            noise_batch = numpy.random.rand(batch_size, dim).astype('float32')
            fake_samples = passThroughGenerativeModel(noise_batch, gen)
            true_n_fake = numpy.concatenate([wanted_digits[first: last],
                                             fake_samples], axis=0)
            y_batch = numpy.concatenate([numpy.ones((batch_size, 1)),
                                         numpy.zeros((batch_size, 1))], axis=0).astype('float32')
            all_fake = numpy.ones((batch_size, 1)).astype('float32')
            if iter % generating_train_percentage == 0 and iter != 0:
                gen_acc += gen.train_on_batch(noise_batch, all_fake)[1]
                gen_count += 1
            else:
                desc_acc += desc.train_on_batch(true_n_fake, y_batch)[1]
                desc_count += 1
        if gen_count != 0:
            gen_acc /= float(gen_count)
            print("Generative accuracy %s" % gen_acc)
        if desc_count != 0:
            desc_acc /= float(desc_count)
            print("Descriptive accuracy %s" % desc_acc)

        fixed_fake = passThroughGenerativeModel(fixed_noise, gen)
        fixed_fake *= 255
        plt.clf()
        plt.imshow(fixed_fake.reshape((28, 28)), cmap='gray')
        plt.axis('off')
        fig.canvas.draw()
        plt.savefig(str(wanted_digit) + "/Iter " + str(iter) + '.png')
        if desc_count != 0 and desc_acc <= 0.5:
            break
    if save_model:
        gen.save_weights(str(wanted_digit) + "/genModel.weights")
        open(str(wanted_digit) + "/genModel.structure", "w").write(gen.to_json())


def passThroughGenerativeModel(x, gen):
    res = []
    for i in range(x.shape[0]):
        res.append(expit(numpy.dot(numpy.maximum(numpy.dot(x[i], gen.layers[0].get_weights()[0]), 0),
                                   gen.layers[2].get_weights()[0])))
    return numpy.array(res)


evaluate(1 / 3, 20, 3000, 9, save_model=True)