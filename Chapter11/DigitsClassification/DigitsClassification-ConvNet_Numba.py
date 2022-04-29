# -*- coding: UTF-8 -*-
""" 
Handwritten Digit Classification in MNIST using Convolutional Neural Network and Numpy/Numba
Ver 0.2 
Code reconstruction using numba
1. Using C-style for-loops
2. Avoiding global variables and pass arguments to static jit methods
"""
import numpy as np
import numba as nb
from tqdm import tqdm
import pickle
import time
#import pdb

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
            #r = np.clip(r, -300, 300)  # clip the output to eliminate overflow error
            return 1 / (1 + np.exp(-r))
        elif activation == 'softmax':  # output layer
            r = r - np.max(r, axis=-1, keepdims=True)  # numerically stable softmax
            exp_r = np.exp(r)
            return exp_r / np.sum(exp_r, axis=-1, keepdims=True)
        return r
    @staticmethod
    def apply_activation_derivative(output, activation=None):
        """ Calculate the derivative of activation function """
        if activation is None: # No activation function, derivative is 1
            return np.ones_like(output)
        elif activation == 'softmax':  # Skip derivation when softmax and categorical cross-entropy loss are chosen
            return np.ones_like(output)
        elif activation == 'relu':
            grad = np.array(output, copy=True)
            grad[output > 0] = 1.
            grad[output <= 0] = 0.
            return grad
        elif activation == 'tanh':
            return 1 - output ** 2
        elif activation == 'sigmoid':
            return output * (1 - output)
        return output

class ConvLayer:
    def __init__(self, n_filters, kernel_size, activation=None, weights=None, bias=None, use_bias=True):
        self.n_filters = n_filters  # n_output_image = n_input_image * n_filters
        self.kernel_size = kernel_size
        self.weights = weights  # filters
        self.bias = bias
        self.activation = activation
        self.use_bias = use_bias
    def initialize(self):  # Xavier initialization
        if self.weights is None:
            self.weights = np.random.randn(self.n_filters, self.kernel_size[0], self.kernel_size[1]) / \
                            np.sqrt(self.kernel_size[0] * self.kernel_size[1])
        if self.bias is None:
            self.bias = np.zeros(self.n_filters)
        self.weights_gradient = np.zeros_like(self.weights)
        self.bias_gradient = np.zeros_like(self.bias)
    @staticmethod
    @nb.jit(nopython=True, fastmath=True)  # accelerate C-style for-loop
    def activate_jit(image, weights, bias, use_bias):
        n_input_image, h, w = image.shape
        n_filters, h_kernel, w_kernel = weights.shape
        r = np.zeros((n_input_image * n_filters, h - h_kernel + 1, w - w_kernel + 1))
        for n in range(n_input_image):
            for i in range(h - h_kernel + 1):
                for j in range(w - w_kernel + 1):
                    for f in range(n_filters):
                        for i_kernel in range(h_kernel):
                            for j_kernel in range(w_kernel):
                                r[n * n_filters + f, i, j] += image[n, i + i_kernel, j + j_kernel] * weights[f, i_kernel, j_kernel]
                        if use_bias:
                            r[n * n_filters + f, i, j] += bias[f]
        return r
    def activate(self, image):
        """ Forward propagation """
        r = self.activate_jit(image, self.weights, self.bias, self.use_bias)
        output = Activate.apply_activation(r, self.activation)
        return output
    @staticmethod
    @nb.jit(nopython=True, fastmath=True)
    def back_propagation_jit(image, weights, bias, delta, use_bias):
        n_input_image, h, w = image.shape
        n_filters, h_kernel, w_kernel = weights.shape
        weights_gradient = np.zeros((n_filters, h_kernel, w_kernel))
        bias_gradient = np.zeros(n_filters)
        error_input = np.zeros((n_input_image, h, w))
        for n in range(n_input_image):
            for i in range(h - h_kernel + 1):
                for j in range(w - w_kernel + 1):
                    for f in range(n_filters):
                        for i_kernel in range(h_kernel):
                            for j_kernel in range(w_kernel):
                                weights_gradient[f, i_kernel, j_kernel] += image[n, i + i_kernel, j + j_kernel] * delta[n * f + f, i, j]
                                error_input[n, i + i_kernel, j + j_kernel] += weights[f, i_kernel, j_kernel] * delta[n * f + f, i, j]
                        if use_bias:
                            bias_gradient[f] += delta[n * f + f, i, j]
        return error_input, weights_gradient, bias_gradient
    def back_propagation(self, image, error, output):
        assert error.shape == output.shape
        delta = error * Activate.apply_activation_derivative(output, self.activation)
        error_input, weights_gradient, bias_gradient = self.back_propagation_jit(image, self.weights, self.bias, delta, self.use_bias)
        self.weights_gradient += weights_gradient
        if self.use_bias:
            self.bias_gradient += bias_gradient
        return error_input

class MaxPoolingLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size
    @staticmethod
    @nb.jit(nopython=True, fastmath=True)  # accelerate C-style for-loop
    def activate_jit(image, pool_size):
        n_input_image, h, w = image.shape
        h_pool = h // pool_size[0]
        w_pool = w // pool_size[1]
        output = np.zeros((n_input_image, h_pool, w_pool))
        for n in range(n_input_image):
            for i in range(h_pool):
                for j in range(w_pool):
                    max_pool = image[n, i*pool_size[0], j*pool_size[1]]
                    for i_pool in range(pool_size[0]):
                        for j_pool in range(pool_size[1]):
                            if max_pool < image[n, i*pool_size[0]+i_pool, j*pool_size[1]+j_pool]:
                                max_pool = image[n, i*pool_size[0]+i_pool, j*pool_size[1]+j_pool]
                    output[n,i,j] = max_pool
        return output
    def activate(self, image):
        output = self.activate_jit(image, self.pool_size)
        return output
    @staticmethod
    @nb.jit(nopython=True, fastmath=True)
    def back_propagation_jit(image, pool_size, error):
        n_input_image, h, w = image.shape
        h_pool = h // pool_size[0]
        w_pool = w // pool_size[1]
        error_input = np.zeros((n_input_image, h, w))
        for n in range(n_input_image):
            for i in range(h_pool):
                for j in range(w_pool):
                    max_pool = image[n, i*pool_size[0], j*pool_size[1]]
                    max_i, max_j = (0, 0)
                    for i_pool in range(pool_size[0]):
                        for j_pool in range(pool_size[1]):
                            if max_pool < image[n, i*pool_size[0]+i_pool, j*pool_size[1]+j_pool]:
                                max_pool = image[n, i*pool_size[0]+i_pool, j*pool_size[1]+j_pool]
                                max_i = i_pool
                                max_j = j_pool
                    error_input[n, i*pool_size[0] + i_pool, j*pool_size[1] + j_pool] = error[n, i, j]
        return error_input
    def back_propagation(self, image, error, output):
        assert error.shape == output.shape
        error_input = self.back_propagation_jit(image, self.pool_size, error)
        return error_input

class FlattenLayer:
    def __init__(self):
        self.image_shape = None
    def activate(self, image):
        self.image_shape = image.shape
        output = image.flatten()
        return output
    def back_propagation(self, image, error, output):
        assert error.shape == output.shape
        error_input = error.reshape(self.image_shape)
        return error_input

class DenseLayer:
    """ A Fully Connected Layer """
    def __init__(self, n_output, activation=None, weights=None, bias=None, use_bias=True):
        self.n_output = n_output
        self.weights = weights
        self.bias = bias
        self.activation = activation  # relu tanh sigmoid or softmax
        self.use_bias = use_bias
    def initialize(self, n_input):  # Xavier initialization
        self.n_input = n_input
        if self.weights is None:
            self.weights = np.random.randn(self.n_input, self.n_output) / np.sqrt(self.n_input)
        if self.bias is None:
            self.bias = np.random.rand(self.n_output) * 0.1
        self.weights_gradient = np.zeros_like(self.weights)
        self.bias_gradient = np.zeros_like(self.bias)
    def activate(self, X):
        """ Forward propagation """
        assert X.ndim == 1
        r = np.dot(X, self.weights)  # X@W + b
        if self.use_bias:
            r += self.bias
        output = Activate.apply_activation(r, self.activation)
        return output
    def back_propagation(self, X, error, output):
        assert error.shape == output.shape
        delta = error*Activate.apply_activation_derivative(output, self.activation)  # calculate delta of X@W + b
        error_input = np.dot(self.weights, delta)  # Chain Rule
        self.weights_gradient += delta * X[np.newaxis].T
        if self.use_bias:
            self.bias_gradient += delta
        return error_input

class DropoutLayer:
    def __init__(self, keep_prob=1):
        self.keep_prob = keep_prob
        self.sample = None
    def activate(self, X):  # apply different binary mask to each example in a minibatch
        assert X.ndim == 1  # dropout is placed after dense layer or flatten layer
        self.sample = np.random.binomial(1, self.keep_prob, X.shape)
        X = X * self.sample
        output = X / self.keep_prob
        return output
    def back_propagation(self, X, error, output):
        assert error.shape == output.shape
        error_input = error * self.sample
        return error_input

class Optimizers:
    def __init__(self, optimizer, learning_rate=0.001, clipvalue=None, decay_rate=0.99, decay_steps=1e4,
                 momentum=0.8, nesterov=True, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.clipvalue = clipvalue
        self.decay_rate, self.decay_steps = decay_rate, decay_steps  # exponential decay of learning rate
        self.momentum, self.nesterov = momentum, nesterov  # SGD
        self.beta_1, self.beta_2, self.epsilon = beta_1, beta_2, epsilon  # ADAM
    def initialize(self, layers, batch_size):
        self.batch_size = batch_size
        #self.learning_rate *= np.sqrt(batch_size)  # scale lr to keep the variance in the gradient expectation constant
        self.n_step = 1
        self.decay_steps = self.decay_steps // batch_size
        if self.optimizer == 'sgd':
            for layer in layers:
                if type(layer) == ConvLayer or type(layer) == DenseLayer:
                    layer.weights_velocity = np.zeros_like(layer.weights)
                    layer.bias_velocity = np.zeros_like(layer.bias)
        elif self.optimizer == 'adam':
            for layer in layers:
                if type(layer) == ConvLayer or type(layer) == DenseLayer:
                    layer.weights_first_moment = np.zeros_like(layer.weights)
                    layer.weights_second_moment = np.zeros_like(layer.weights)
                    layer.bias_first_moment = np.zeros_like(layer.bias)
                    layer.bias_second_moment = np.zeros_like(layer.bias)
        else:
            raise ValueError("Unknown Optimizer")
    def apply(self, layers):
        for layer in layers:
            if type(layer) == ConvLayer or type(layer) == DenseLayer:
                layer.weights_gradient /= self.batch_size  # averaging gradients over batch
                layer.bias_gradient /= self.batch_size
                if self.clipvalue is not None:  # avoid exploding gradients with gradient clipping
                    layer.weights_gradient = np.clip(layer.weights_gradient, -self.clipvalue, self.clipvalue)
                    layer.bias_gradient = np.clip(layer.bias_gradient, -self.clipvalue, self.clipvalue)
                if self.optimizer == 'sgd':
                    layer.weights, layer.weights_velocity = self.sgd_jit(layer.weights, layer.weights_gradient, layer.weights_velocity, 
                                                                         self.momentum, self.learning_rate, self.nesterov)
                    layer.bias, layer.bias_velocity = self.sgd_jit(layer.bias, layer.bias_gradient, layer.bias_velocity, 
                                                                   self.momentum, self.learning_rate, self.nesterov)
                elif self.optimizer == 'adam':
                    layer.weights, layer.weights_first_moment, layer.weights_second_moment = self.adam_jit(layer.weights, layer.weights_gradient, layer.weights_first_moment, layer.weights_second_moment, 
                                                                                                           self.beta_1, self.beta_2, self.epsilon, self.n_step, self.learning_rate)
                    layer.bias, layer.bias_first_moment, layer.bias_second_moment = self.adam_jit(layer.bias, layer.bias_gradient, layer.bias_first_moment, layer.bias_second_moment, 
                                                                                                  self.beta_1, self.beta_2, self.epsilon, self.n_step, self.learning_rate)
                layer.weights_gradient = np.zeros_like(layer.weights)
                layer.bias_gradient = np.zeros_like(layer.bias)
        if self.n_step % self.decay_steps == 0:
            self.learning_rate *= self.decay_rate
        self.n_step += 1
    @staticmethod
    @nb.jit(nopython=True, fastmath=True)
    def sgd_jit(weights, gradient, velocity, momentum, learning_rate, nesterov):
        """ Stochastic dradient Descent (with momentum) """
        velocity = momentum * velocity + gradient
        if nesterov:
            weights -= (momentum * velocity + gradient) * learning_rate
        else:
            weights -= velocity * learning_rate
        return weights, velocity
    @staticmethod
    @nb.jit(nopython=True, fastmath=True)
    def adam_jit(weights, gradient, first_moment, second_moment, beta_1, beta_2, epsilon, n_step, learning_rate):
        """ Adaptive moment estimation """
        first_moment = beta_1 * first_moment + (1-beta_1) * gradient
        second_moment = beta_2 * second_moment + (1-beta_2) * gradient * gradient
        first_unbias = first_moment / (1-beta_1**n_step)
        second_unbias = second_moment / (1-beta_2**n_step)
        weights -= first_unbias / (np.sqrt(second_unbias) + epsilon) * learning_rate
        return weights, first_moment, second_moment

class NeuralNetwork:
    def __init__(self):
        self._layers = []
        self._mses = []  # Mean Square Errors on training set
        self._ces = []  # Cross Entropy on training set
        self._accuracy = []  # Accuracy on test set
    def add_layer(self, layer):
        self._layers.append(layer)
    def feed_forward(self, X):
        """ Forward propagation """
        assert X.ndim == 2
        X = np.expand_dims(X, axis=0)  # X.shape = (1, h ,w)
        output = []
        for i in range(len(self._layers)):
            layer = self._layers[i]
            X = layer.activate(X)
            output.append(X)
            if __debug__:
                if np.max(np.abs(X)) > 2e2:
                    #pdb.set_trace()
                    print('Exploding Output at layer %s' %(i))  # lr or gradients too high
        return output  # outputs of each layer
    def backpropagation(self, X, y):
        output = self.feed_forward(X)
        loss = output[-1] - y  # gradients of categorical cross-entropy loss with softmax classification
        error = loss
        for i in reversed(range(len(self._layers))):  # propagate the error backward
            layer = self._layers[i]
            X_input = np.expand_dims(X, axis=0) if i == 0 else output[i - 1]
            error = layer.back_propagation(X_input, error, output[i])
            if __debug__:
                if np.max(np.abs(error)) > 1e1:
                    #pdb.set_trace()
                    print('Exploding Loss at layer %s' %(i))  # lr or gradients too high
        return loss
    def initialize(self, X):
        """ initialize parameters of layers """
        X = np.expand_dims(X, axis=0)
        for layer in self._layers:
            if type(layer) == ConvLayer:
                layer.initialize()
            if type(layer) == DenseLayer:
                layer.initialize(len(X))
            X = layer.activate(X)
    def train(self, X_train, X_test, y_train, y_test, batch_size, max_epochs, optimizer):
        assert X_train.ndim == 3  # X_train.shape = (n_image, h ,w)
        self.initialize(X_train[0])
        optimizer.initialize(self._layers, batch_size)
        y_onehot = np.zeros((y_train.shape[0], 10))  # one-hot coding
        y_onehot[np.arange(y_train.shape[0]), y_train] = 1
        mse_batch = 0.0
        for i in range(max_epochs):
            pbar = tqdm(total=len(X_train), desc='Batch MSE: %.4f'%(mse_batch), bar_format='{l_bar}{bar:40}| {n_fmt}/{total_fmt} {elapsed}')
            for j in range(len(X_train)):
                loss = self.backpropagation(X_train[j], y_onehot[j])
                mse_batch += np.mean(np.square(loss))
                if j % batch_size == batch_size - 1:
                    pbar.set_description('Batch MSE: %.4f'%(mse_batch / batch_size))
                    pbar.update(batch_size)
                    mse_batch = 0.0
                    optimizer.apply(self._layers)
            pbar.close()
            y_predict = self.predict(X_train)
            mse = np.mean(np.square(y_predict - y_onehot))
            self._mses.append(mse)
            ce = np.mean(-np.sum(y_onehot * np.log(y_predict), axis = 1))
            self._ces.append(ce)
            accuracy = self.accuracy(self.predict(X_test), y_test.flatten())
            self._accuracy.append(accuracy)
            print('Epoch: #%s, MSE: %.6f, CE: %.6f, Accuracy: %.2f%%' %(i+1, mse, ce, accuracy * 100))
            self.save('model.pkl')  # saving checkpoint
    def accuracy(self, y_predict, y_test):
        y_predict = np.argmax(y_predict, axis=1)
        return np.sum(y_predict == y_test) / len(y_test)
    def predict(self, X_predict):
        if X_predict.ndim == 2:
            X_predict = np.expand_dims(X_predict, axis=0)
        y_predict = np.zeros((X_predict.shape[0], 10))
        for i in range(len(X_predict)):
            X = np.expand_dims(X_predict[i], axis=0)
            for layer in self._layers:
                if type(layer) == DropoutLayer:  # skip dropout layer
                    continue
                X = layer.activate(X)
            y_predict[i] = X
        return y_predict
    def save(self, file_name):  # saving trained model to disk
        f = open(file_name, 'wb')
        pickle.dump(self, f)
        f.close()
    def read(self, file_name):  # reading trained model from disk
        try:
            print('Reading trained model from %s' %(file_name))
            f= open(file_name, 'rb')
            tmp_dict = pickle.load(f)
            f.close()
            for i in range(len(self.__dict__['_layers'])):
                if type(self._layers[i]) != type(tmp_dict._layers[i]):
                    print('Different network found in %s' %(file_name))
                    return
                elif type(self._layers[i]) == ConvLayer or type(self._layers[i]) == DenseLayer:
                    self._layers[i].weights = tmp_dict._layers[i].weights
                    self._layers[i].bias = tmp_dict._layers[i].bias
        except IOError: print('File %s is not accessible' %(file_name))

try:
    f = np.load('mnist.npz')
    X_train, y_train = f['x_train'], f['y_train']
    X_test, y_test = f['x_test'], f['y_test'] 
    f.close()
except IOError: print('mnist.npz is not accessible')

X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

nn = NeuralNetwork()
nn.add_layer(ConvLayer(n_filters=8, kernel_size=(5, 5), activation='relu'))  # 1 image (28,28) => 8 image (24,24)
nn.add_layer(MaxPoolingLayer(pool_size=(2, 2)))  # 8 image (24,24) =>  8 image (12,12)
nn.add_layer(ConvLayer(n_filters=8, kernel_size=(3, 3), activation='relu'))  # 8 image (12,12) => 64 image (10,10)
nn.add_layer(MaxPoolingLayer(pool_size=(2, 2)))  # 64 image (10,10) =>  64 image (5,5)
nn.add_layer(FlattenLayer()) # 64 image (5,5) => 1600 pixels
nn.add_layer(DenseLayer(n_output=100, activation='sigmoid'))  # 1600 => 100
nn.add_layer(DropoutLayer(keep_prob=0.6))  # randomly shut down 40% neurons
nn.add_layer(DenseLayer(n_output=10, activation='softmax'))  # output layer, 100 => 10

nn.read('model.pkl')  # reading checkpoints if model.pkl exists

time_start = time.time()
opt = Optimizers(optimizer='sgd', learning_rate=0.002)
nn.train(X_train, X_test, y_train, y_test, batch_size=50, max_epochs=100, optimizer=opt)
time_end = time.time()
print("Times Used %.2f S"%(time_end - time_start))
