""" 
Handwritten Digit Classification using Convolutional Neural Network and Numpy
MNIST
"""
import numpy as np
import pickle
from progress.bar import IncrementalBar
import time

class Activate:
    @staticmethod
    def apply_activation(r, activation=None):
        if activation is None:
            return r
        elif activation == 'relu':
            return np.maximum(r, 0)
        elif activation == 'tanh':
            return np.tanh(r)
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))
        elif activation == 'softmax':
            return np.exp(r) / np.sum(np.exp(r), axis=-1, keepdims=True)
        return r
    @staticmethod
    def apply_activation_derivative(output, activation=None):
        # Calculate the derivative of activation function
        if activation is None: # No activation function, derivative is 1
            return np.ones_like(output)
        # ReLU
        elif activation == 'relu':
            grad = np.array(output, copy=True)
            grad[output > 0] = 1.
            grad[output <= 0] = 0.
            return grad
        # Tanh
        elif activation == 'tanh':
            return 1 - output ** 2
        # Sigmoid
        elif activation == 'sigmoid':
            return output * (1 - output)
        return output

class ConvLayer:
    def __init__(self, n_filters, kernel_size=3, activation=None):
        self.n_filters = n_filters  # n_output_image = n_input_image * n_filters
        self.kernel_size = kernel_size  # Square filter
        self.filters = np.random.randn(n_filters, kernel_size, kernel_size) / kernel_size
        self.filters_update = np.zeros_like(self.filters)
        self.activation = activation
        self.n_input_image = None
        self.output = None
        self.error = None  # Error or loss gradient for the output
        self.error_input = None
    def iterate_regions(self, image):
        # Generates all possible image regions using valid padding
        h, w = image.shape[-2:]
        for n in range(self.n_input_image):
            for i in range(h - self.kernel_size + 1):
                for j in range(w - self.kernel_size + 1):
                    im_region = image[n, i:(i + self.kernel_size), j:(j + self.kernel_size)]
                    yield im_region, n, i, j
    def activate(self, image):
        # Forward propagation
        if image.ndim == 2:  # if layer is input layer
            image = np.expand_dims(image, axis=0)  # image.shape = (n, h ,w)
        self.n_input_image, h, w = image.shape
        r = np.zeros((self.n_input_image * self.n_filters, h - self.kernel_size + 1, w - self.kernel_size + 1))
        for im_region, n, i, j in self.iterate_regions(image):
            r[n * self.n_filters : (n+1) * self.n_filters, i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        self.output = Activate.apply_activation(r, self.activation)
        return self.output
    def back_propagation(self, image, error, learning_rate):
        assert error.shape == self.output.shape
        self.error = error
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)  # image.shape = (n, h ,w)
        self.error_input = np.zeros(image.shape)
        self.delta = self.error * Activate.apply_activation_derivative(self.output, self.activation)
        for im_region, n, i, j in self.iterate_regions(image):
            for f in range(self.n_filters):
                self.filters_update[f] -=  im_region * self.delta[n * f + f, i, j] * learning_rate  # gradient descent
                self.error_input[n, i:(i + self.kernel_size), j:(j + self.kernel_size)] += self.filters[f] * self.delta[n * f + f, i, j]

class MaxPoolingLayer:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size
        self.n_input_image = None
        self.output = None
        self.error = None  # Error or loss gradient for the output
        self.error_input = None
    def iterate_regions(self, image):
        # Generates all possible image regions using valid padding
        h, w = image.shape[-2:]
        h_pool = h // self.pool_size
        w_pool = w // self.pool_size
        for n in range(self.n_input_image):
            for i in range(h_pool):
                for j in range(w_pool):
                    im_region = image[n, i*self.pool_size : (i + 1)*self.pool_size, j*self.pool_size : (j + 1)*self.pool_size]
                    yield im_region, n, i, j
    def activate(self, image):
        if image.ndim == 2:  # if layer is input layer
            image = np.expand_dims(image, axis=0)  # image.shape = (n, h ,w)
        self.n_input_image, h, w = image.shape
        self.output = np.zeros((self.n_input_image, h//self.pool_size, w//self.pool_size))
        for im_region, n, i, j in self.iterate_regions(image):
            self.output[n,i,j] = np.max(im_region)
        return self.output
    def back_propagation(self, image, error, learning_rate):
        assert error.shape == self.output.shape
        self.error = error
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)  # image.shape = (n, h ,w)
        self.error_input = np.zeros(image.shape)
        for im_region, n, i, j in self.iterate_regions(image):
            i_pool, j_pool = np.unravel_index(im_region.argmax(), im_region.shape)
            self.error_input[n, i*self.pool_size + i_pool, j*self.pool_size + j_pool] = self.error[n, i, j]

class FlattenLayer:
    def __init__(self):
        self.output = None
        self.image_shape = None
        self.error = None
        self.error_input = None
    def activate(self, image):
        self.image_shape = image.shape
        self.output = image.flatten()
        return self.output
    def back_propagation(self, image, error, learning_rate):
        assert error.shape == self.output.shape
        self.error = error
        self.error_input = self.error.reshape(self.image_shape)

class DenseLayer:
    """ A Fully Connected Layer """
    def __init__(self, n_input, n_output, activation=None, weights=None, bias=None):
        """
        :param int n_input: Number of input nodes/neurons of previous layer
        :param int n_output: Number of output nodes/neurons of this layer
        :param str activation: Type of activation function
        :param weights : Weight of input connections
        :param bias : Bias of input connections
        """
        self.weights = weights if weights is not None else np.random.randn(n_input, n_output) / np.sqrt(n_input) # Xavier initialization
        self.bias = bias if bias is not None else np.random.rand(n_output) * 0.2
        self.weights_update = np.zeros_like(self.weights)  # weights_new = weights_old + weights_update
        self.bias_update = np.zeros_like(self.bias)  # bias_new = bias_old + bias_update
        self.activation = activation  # relu tanh or sigmoid
        self.output = None  # Output/activation value of this layer
        self.n_input = n_input
        self.n_output = n_output
        self.error = None  # Error or loss gradient of output, error = y - output
        self.delta = None  # Delta of X@W + b, delta = error*activation_derivative(output)
        self.error_input = None  # Error or loss gradient of input, previous_layer.error = layer.error_input
    def activate(self, X):
        assert X.ndim == 1
        assert len(X) == self.n_input
        # Forward propagation
        r = np.dot(X, self.weights) + self.bias # X@W + b
        assert max(r) < 1e3, 'Exploding Output'
        self.output = Activate.apply_activation(r, self.activation)
        return self.output
    def back_propagation(self, X, error, learning_rate):
        assert error.shape == self.output.shape
        self.error = error
        # Calculate delta of layer
        if self.activation == 'softmax':  # output layer
            self.delta = self.error
        else:
            self.delta = self.error*Activate.apply_activation_derivative(self.output, self.activation)
        self.error_input = np.dot(self.weights, self.delta)  # Chain Rule
        # Calculate weight_update and bias_update
        X = np.atleast_2d(X)  # prepare for transpose
        self.weights_update -= self.delta * X.T * learning_rate  # gradient descent
        self.bias_update -= self.delta * learning_rate

class DropoutLayer:
    def __init__(self, keep_prob=1):
        self.keep_prob = keep_prob
        self.sample = None
        self.output = None
        self.error = None
        self.error_input = None
    def activate(self, X):
        assert X.ndim == 1  # dropout is placed on the fully connected layers only
        self.sample = np.random.binomial(1, self.keep_prob, X.shape)
        X = X * self.sample
        self.output = X / self.keep_prob
        return self.output
    def back_propagation(self, X, error, learning_rate):
        assert error.shape == self.output.shape
        self.error = error
        self.error_input = self.error * self.sample

class NeuralNetwork:
    def __init__(self):
        self._layers = []
        self._mses = []  # Mean Square Errors on training set
        self._ces = []  # Cross Entropy on training set
        self._accuracy = []  # Accuracy on test set
    def add_layer(self, layer):
        self._layers.append(layer)

    def feed_forward_train(self, X):
        # Forward propagation
        assert X.ndim == 2
        for layer in self._layers:
            X = layer.activate(X)
        return X
    def feed_forward(self, X):
        """ Forward propagation """
        assert X.ndim == 2
        for layer in self._layers:
            if type(layer) == DropoutLayer:  # skip dropout layer
                continue
            X = layer.activate(X)
        return X      
    def backpropagation(self, X, y, learning_rate):
        output = self.feed_forward_train(X)
        for i in reversed(range(len(self._layers))): # propagate the error backward
            layer = self._layers[i]
            X_input = X if i == 0 else self._layers[i - 1].output
            if layer == self._layers[-1]: # output layer
                error = output - y  # Softmax classification with categorical cross-entropy loss
                layer.back_propagation(X_input, error, learning_rate)
            else: # hidden layer
                error = self._layers[i + 1].error_input
                layer.back_propagation(X_input, error, learning_rate)
            assert np.max(error) < 1e1, 'Exploding Gradient'
    def train(self, X_train, X_test, y_train, y_test, learning_rate, batch_size, max_epochs):
        y_onehot = np.zeros((y_train.shape[0], 10))  # one-hot coding
        y_onehot[np.arange(y_train.shape[0]), y_train] = 1
        for i in range(1, max_epochs+1):
            bar = IncrementalBar('Processing', max=len(X_train), suffix='%(index)d/%(max)d %(elapsed)ds')
            for j in range(len(X_train)):
                self.backpropagation(X_train[j], y_onehot[j], learning_rate)  # one sample each train
                if j % batch_size == batch_size - 1:  # averaging updates over batch
                    bar.next(batch_size)
                    for k in range(len(self._layers)):
                        layer = self._layers[k]
                        if type(layer) == ConvLayer:
                            layer.filters += layer.filters_update / batch_size
                            layer.filters_update = np.zeros_like(layer.filters)
                        if type(layer) == DenseLayer:
                            layer.weights += layer.weights_update / batch_size
                            layer.bias += layer.bias_update / batch_size
                            layer.weights_update = np.zeros_like(layer.weights)
                            layer.bias_update = np.zeros_like(layer.bias)
            bar.finish()
            #self.save('model.pkl')  # save model for each epoch
            learning_rate = learning_rate * 0.95  # decay learning rate
            # print Mean Square Error and Cross Entropy Loss
            y_predict = self.predict(X_train)
            mse = np.mean(np.square(y_predict - y_onehot))
            self._mses.append(mse)
            ce = np.mean(-np.sum(y_onehot * np.log(y_predict), axis = 1))
            self._ces.append(ce)
            accuracy = self.accuracy(self.predict(X_test), y_test.flatten())
            self._accuracy.append(accuracy)
            print('Epoch: #%s, MSE: %f, CE: %f, Accuracy: %.2f%%' %(i, mse, ce, accuracy * 100))
    def accuracy(self, y_predict, y_test):
        y_predict = np.argmax(y_predict, axis=1)
        return np.sum(y_predict == y_test) / len(y_test)
    def predict(self, X_predict):
        X_predict = np.atleast_2d(X_predict)
        y_predict = np.zeros((X_predict.shape[0], 10))
        for i in range(len(X_predict)):
            y_predict[i] = self.feed_forward(X_predict[i]) #  probability distribution of y_predict
        return y_predict
    def save(self, file_name):
        f = open(file_name, 'wb')
        pickle.dump(self, f)
        f.close()
    def read(self, file_name):
        f= open(file_name, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.clear()
        self.__dict__.update(tmp_dict.__dict__)

f = np.load('mnist.npz')
X_train, y_train = f['x_train'], f['y_train']
X_test, y_test = f['x_test'], f['y_test'] 
f.close()

X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

nn = NeuralNetwork()
nn.add_layer(ConvLayer(n_filters=4, kernel_size=3, activation='relu'))  # 1 image (28,28) => 4 image (26,26)
nn.add_layer(MaxPoolingLayer(pool_size=2))  # 4 image (26,26) =>  4 image (13,13)
nn.add_layer(ConvLayer(n_filters=4, kernel_size=3, activation='relu'))  # 4 image (13,13) => 16 image (11,11)
nn.add_layer(MaxPoolingLayer(pool_size=2))  # 16 image (11,11) =>  16 image (5,5)
nn.add_layer(FlattenLayer()) # 16 image (5,5) => 400 pixels
nn.add_layer(DropoutLayer(keep_prob=0.6))  # randomly shut down 40% neurons
nn.add_layer(DenseLayer(4*4*5*5, 10, 'softmax'))  # output layer, 676 => 10

# Mini-Batch Gradient Descent
batch_size = 10
learning_rate = 0.01 * np.sqrt(batch_size)
time_start = time.time()
nn.train(X_train[:200], X_test[:200], y_train[:200], y_test[:200], learning_rate, batch_size, max_epochs=2)
time_end = time.time()
print("Times Used %.2f S"%(time_end - time_start))
