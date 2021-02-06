# Imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.nn import relu, softmax


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


x_train = x_train / 255
x_test = x_test / 255
x_train = tf.expand_dims(x_train, -1)
x_test = tf.expand_dims(x_test, -1)


model = Sequential()
model.add(Conv2D(32, (3,3), activation=relu, input_shape=x_train[0].shape))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation=relu))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation=relu))

model.add(Flatten())
model.add(Dense(128, activation=relu))
model.add(Dense(10, activation=softmax))


model.compile(optimizer="adam",
             loss="sparse_categorical_crossentropy",
             metrics=["accuracy"])


model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))


# model.save("cnn.model")


loss, acc = model.evaluate(x_test, y_test)
print(f"loss - {loss}")
print(f"accuracy - {acc}")
