""" 
Binary Classification 二分类 
https://www.cnblogs.com/jsfantasy/p/12177216.html
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

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
                # calculate delta of MSE cost
                #layer.delta = layer.error * layer.apply_activation_derivative(output)
                # calculate delta of cross-entropy cost
                layer.delta = layer.error
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
        # one-hot coding
        y_onehot = np.zeros((y_train.shape[0], 2))
        y_onehot[np.arange(y_train.shape[0]), y_train] = 1
        mses = []  # Mean square errors
        for i in range(max_epochs):
            for j in range(len(X_train)):  # one sample each train
                self.backpropagation(X_train[j], y_onehot[j], learning_rate)
                if j % batch_size == batch_size - 1:
                    for k in range(len(self._layers)):
                        layer = self._layers[k]
                        layer.weights += layer.weights_update / batch_size  # averaging over this batch
                        layer.bias += layer.bias_update / batch_size
                        layer.weights_update = np.zeros_like(layer.weights)  # resatrt averaging over this batch
                        layer.bias_update = np.zeros_like(layer.bias)
            if i % 10 == 0:
                # print MSE Loss
                mse = np.mean(np.square(y_onehot - self.feed_forward(X_train)))
                mses.append(mse)
                print('Epoch: #%s, MSE: %f, Accuracy: %.2f%%' %
                    (i, float(mse), self.accuracy(self.predict(X_test), y_test.flatten()) * 100))
        return mses
    def accuracy(self, y_predict, y_test):
        return np.sum(y_predict == y_test) / len(y_test)
    def predict(self, X_predict):
        y_predict = self.feed_forward(X_predict) #  probability distribution of y_predict
        y_predict = np.argmax(y_predict, axis=1) #  find the index/class of highest probability
        return y_predict

X, y = datasets.make_moons(n_samples=1000, noise=0.2, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

nn = NeuralNetwork()  # 3 layers, 2 inputs, 2 outputs
nn.add_layer(Layer(2, 20 , 'sigmoid'))  # hidden layer 1, 2 input => 20 output
nn.add_layer(Layer(20, 20, 'sigmoid'))  # hidden layer 2, 20 => 20
nn.add_layer(Layer(20, 2, 'sigmoid'))  # output layer, 20 => 2

nn.train(X_train, X_test, y_train, y_test, learning_rate=0.01, max_epochs=101, batch_size=1)  # online learning

def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0])*100)).reshape(1, -1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2])*100)).reshape(-1, 1)
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predic = model.predict(X_new)
    zz = y_predic.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF590', '#90CAF9'])
    plt.contourf(x0, x1, zz, cmap=custom_cmap)

plt.figure(figsize=(12, 8)) 
plot_decision_boundary(nn, [-2, 2.5, -1, 2])
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
#plt.show()
plt.savefig("BinaryClassification.png", format="png", dpi=150)
