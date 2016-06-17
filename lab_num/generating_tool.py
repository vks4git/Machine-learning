import numpy
from keras.models import model_from_json
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import os
from main_gal import passThroughGenerativeModel


def generate(wanted_digit, dim, x=None, fname=""):
    if not os.path.exists(str(wanted_digit) + "/"):
        print("Sorry, no generating model has been trained yet. Try another digit.")
        return
    gen = model_from_json(open(str(wanted_digit) + "/genModel.structure", 'r').read())
    gen.load_weights(str(wanted_digit) + "/genModel.weights")
    gen.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True),
                metrics=['accuracy'])
    if x is None:
        x = numpy.random.rand(1, dim).astype('float32')
    res = passThroughGenerativeModel(x, gen)
    fig = plt.figure()
    plt.clf()
    plt.imshow(res.reshape((28, 28)), cmap='gray')
    plt.axis('off')
    fig.canvas.draw()
    if fname == "":
        plt.savefig(str(wanted_digit) + "/generatedSample.png")
    else:
        plt.savefig(str(wanted_digit) + "/" + fname + ".png")