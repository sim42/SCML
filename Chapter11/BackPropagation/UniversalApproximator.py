"""
Universal Function Approximators 函数拟合器
Multi-layer Perceptrons (NN) is Universal Approximators provided sufficiently broad and deep
http://deeplearning.cs.cmu.edu/F20/index.html
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Layer:
    """ A Fully Connected Layer """
    def __init__(self, n_input, n_output, activation=None, weights=None, bias=None):
        """
        :param int n_input: Number of input nodes/neurons of previous layer
        :param int n_output: Number of output nodes/neurons of this layer
        :param str activation: Type of activation function
        :param weights : Weight of input connections
        :param bias : Bias of input connections
        """
        self.weights = weights if weights is not None else np.random.randn(n_input, n_output) / np.sqrt(n_input) * 2 # Normalization
        self.bias = bias if bias is not None else np.random.rand(n_output) * 0.2
        self.weights_update = np.zeros_like(self.weights)  # weights_new = weights_old + weights_update
        self.bias_update = np.zeros_like(self.bias)  # bias_new = bias_old + bias_update
        self.activation = activation  # relu tanh or sigmoid
        self.activation_output = None  # Output/activation value of this layer
        self.error = None  # Error or discrepancy of output/activation value
        self.delta = None  # Delta of X@W + b, delta = error*activation_derivative(output)
    def activate(self, X):
        # Forward propagation function
        r = np.dot(X, self.weights) + self.bias # X@W + b
        # Output of fully connected layer, o (activation_output)
        self.activation_output = self._apply_activation(r)
        return self.activation_output
    def _apply_activation(self, r):
        if self.activation is None:
            return r
        elif self.activation == 'relu':
            return np.maximum(r, 0)
        elif self.activation == 'tanh':
            return np.tanh(r)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))
        return r
    def apply_activation_derivative(self, y):
        # Calculate the derivative of activation function
        if self.activation is None: # No activation function, derivative is 1
            return np.ones_like(y)
        # ReLU
        elif self.activation == 'relu':
            grad = np.array(y, copy=True)
            grad[y > 0] = 1.
            grad[y <= 0] = 0.
            return grad
        # Tanh
        elif self.activation == 'tanh':
            return 1 - y ** 2
        # Sigmoid
        elif self.activation == 'sigmoid':
            return y * (1 - y)
        return y

class NeuralNetwork:
    def __init__(self):
        self._layers = []
    def add_layer(self, layer):
        self._layers.append(layer)
    def feed_forward(self, X):
        # Forward propagation
        for layer in self._layers:
            X = layer.activate(X)
        return X
    def backpropagation(self, X, y, learning_rate):
        # Calculate delta of each layer
        output = self.feed_forward(X)
        for i in reversed(range(len(self._layers))): # reverse looping
            layer = self._layers[i]
            if layer == self._layers[-1]: # output layer
                layer.error = y - output
                # calculate delta of final layer
                layer.delta = layer.error * layer.apply_activation_derivative(output)
            else: # hidden layer
                next_layer = self._layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error*layer.apply_activation_derivative(layer.activation_output)
        # Calculate weight_update and bias_update
        for i in range(len(self._layers)):
            layer = self._layers[i]
            # o_i is the output/activation value of previous layer
            o_i = np.atleast_2d(X if i == 0 else self._layers[i - 1].activation_output)
            layer.weights_update += layer.delta * o_i.T * learning_rate
            layer.bias_update += layer.delta * learning_rate
    def train(self, X_train, X_test, y_train, y_test, learning_rate, max_epochs, batch_size):
        mses = []  # Mean square errors
        for i in range(max_epochs):
            for j in range(len(X_train)):  # one sample each train
                self.backpropagation(X_train[j], y_train[j], learning_rate)
                if j % batch_size == batch_size - 1:  # averaging over this batch
                    for k in range(len(self._layers)):
                        layer = self._layers[k]
                        layer.weights += layer.weights_update / batch_size
                        layer.bias += layer.bias_update / batch_size
                        layer.weights_update = np.zeros_like(layer.weights)  # resatrt averaging over this batch
                        layer.bias_update = np.zeros_like(layer.bias)
            if i % 10 == 0:
                # print MSE Loss
                mse = np.mean(np.square(y_train - self.feed_forward(X_train)))
                mse_test = np.mean(np.square(y_test - self.feed_forward(X_test)))
                mses.append(mse)
                print('Epoch: #%s, Train MSE: %f, Test MSE: %f' %(i, float(mse), float(mse_test)))
        return mses

def f(X):
    """
    y = sin(x0) + sin(x1)
    """
    return np.sum(np.sin(X), axis = 1)

num_train = 200
num_test = 20
X_train = np.random.rand(num_train,2) * 5  # x1 x2 [0-5]
y_train = f(X_train).reshape(num_train,1) + 0.1 * np.random.rand(num_train,1)
X_test = np.random.rand(num_test,2) * 5
y_test = f(X_test).reshape(num_test,1)

nn = NeuralNetwork()  # 3 layers, 2 inputs, 1 outputs
nn.add_layer(Layer(2, 20 , 'sigmoid'))  # hidden layer 1, 2 input => 20 output
nn.add_layer(Layer(20, 20, 'sigmoid'))  # hidden layer 2, 20 => 20
nn.add_layer(Layer(20, 1))  # output layer, 20 => 1, no activation

nn.train(X_train, X_test, y_train, y_test, learning_rate=0.1, max_epochs=201, batch_size=1)

n = 500
x0, x1 = np.meshgrid(np.linspace(0, 8, n), np.linspace(0, 8, n))
y = np.sin(x0) + np.sin(x1)
X_new = np.c_[x0.ravel(), x1.ravel()]
y_predict = nn.feed_forward(X_new).reshape(n,n)

fig = plt.figure(figsize=(8, 8))
ax = Axes3D(fig)
ax.plot_wireframe(x0, x1, y, rstride=20,
                  cstride=20, linewidth=0.5,
                  color='blue')
ax.plot_wireframe(x0, x1, y_predict, rstride=20,
                  cstride=20, linewidth=0.5,
                  color='orangered')
ax.plot_surface(x0, x1, y_predict-y-2, cmap=plt.get_cmap('rainbow'))
#plt.show()
plt.savefig("UniversalApproximator.png", format="png", dpi=150)
