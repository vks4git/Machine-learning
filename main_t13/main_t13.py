from keras.datasets import imdb
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import LSTM
from keras.layers import Dense, Activation
from keras.layers import Embedding

length = 100
nb_epoch = 3
batch_size = 100
features = 20000

(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=features,
                                                      test_split=0.2)

X_train = sequence.pad_sequences(X_train, maxlen=length)
X_test = sequence.pad_sequences(X_test, maxlen=length)

model = Sequential()
model.add(Embedding(features, 128, input_length=length))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print("Score: %s" % score[1])
