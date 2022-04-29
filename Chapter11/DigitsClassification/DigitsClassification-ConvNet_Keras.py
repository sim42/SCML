""" 
Handwritten Digit Classification using Convolutional Neural Network and Keras
MNIST
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import time

(X_train, y_train), (X_test, y_test) = mnist.load_data("mnist.npz")

# Scale images to the [0, 1] range
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

num_classes = 10
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

input_shape = (28, 28, 1)
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(8, kernel_size=(5, 5), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.6),
        #layers.Dense(100, activation="sigmoid"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
# Adaptive Moment Estimation
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9), metrics=["accuracy"])

batch_size = 20
epochs = 10
time_start = time.time()
model.fit(X_train[:10000], y_train[:10000], batch_size=batch_size, epochs=epochs, validation_split=0.1)
time_end = time.time()
print("Times Used %.2f S"%(time_end - time_start))

score = model.evaluate(X_test[:10000], y_test[:10000], verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

