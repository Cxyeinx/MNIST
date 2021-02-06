import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


print(f"X train shape {x_train.shape}")
print(f"Y train shape {y_train.shape}")
print(f"X test shape {x_test.shape}")
print(f"Y test shape {y_test.shape}")


print(x_train[0])
plt.imshow(x_train[0],cmap=plt.cm.binary)
print(y_train[0])


x_test = x_test / 255
x_train = x_train / 255


print(x_train[0])
plt.imshow(x_train[0],cmap=plt.cm.binary)
print(y_train[0])


model = keras.models.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))


model.compile(optimizer="adam",
             loss="sparse_categorical_crossentropy",
             metrics=['accuracy'])


model.fit(x_train, y_train, epochs=3, validation_split=0.2)


# model.save("mnist.model")
# model = tf.keras.models.load_model('model.model')



loss, acc = model.evaluate(x_test, y_test)
print(f"loss - {loss}")
print(f"accuracy - {acc * 100}get_ipython().run_line_magic("")", "")


test = x_test[100]
print(test.shape)
test = test.reshape(1, 28, 28)


prediction = model.predict(test)


print(np.argmax(prediction))
print(y_test[100])
plt.imshow(x_test[100],cmap=plt.cm.binary)






