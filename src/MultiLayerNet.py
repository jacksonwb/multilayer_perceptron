#! /usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  MultiLayerNet.py                                                            #
#                                                                              #
#  By - jacksonwb                                                              #
#  Created: Wednesday December 1969 4:00:00 pm                                 #
#  Modified: Monday Sep 2019 9:15:31 pm                                        #
#  Modified By: jacksonwb                                                      #
# ---------------------------------------------------------------------------- #

import numpy as np

class ConnectedLayer:
	def __init__(self, input_n, size):
		self.input_n = input_n
		self.size = size
		self.w = 0.01 * np.random.randn(input_n, size)
		self.b = 0.01 * np.ones([1, size])
	def forward(self, x):
		return np.dot(x, self.w) + self.b
	def backward(self, x, grad_out):
		dw = np.dot(x.T, grad_out)
		self.dw = dw
		self.db = np.sum(grad_out, axis=0, keepdims=True)
		return np.dot(grad_out, self.w.T)
	def update(self, eta):
		self.w += -eta * self.dw
		self.b += -eta * self.db

class ReLU:
	def __init__(self, input_n, size):
		self.size = size
		self.input_n = input_n
	def forward(self, x):
		return(np.maximum(0, x))
	def backward(self, x, grad_out):
		grad_out[x <= 0 ] = 0
		return grad_out
	def update(self, eta):
		pass

class SoftmaxCrossEntropy:
	def forward(self, x):
		exp = np.exp(x)
		return exp / np.sum(exp, axis=1, keepdims=True)
	def backward(self, y, y_pred, grad_out):
		p = y_pred.copy()
		p[range(y.shape[0]), y] -= 1
		p /= y.shape[0]
		return p * grad_out
	def cost(self, y, y_pred):
		n = y.shape[0]
		y_diff = y_pred[range(n), y]
		log_likelihood = -np.log(y_diff)
		loss = np.sum(log_likelihood) / n
		return loss

layer_types = {'ReLU':ReLU, 'connected':ConnectedLayer}

class Network:
	def __init__(self):
		self.trained = False
		self.layers = []
		self.output_layer = SoftmaxCrossEntropy()
		self.loss = []
		self.val_loss = []
	def add_layer(self, type, input_n, size):
		self.layers.append(layer_types[type](input_n, size))
	def forward(self, x):
		self.x = [x]
		for layer in self.layers:
			x = layer.forward(x)
			self.x.append(x)
		return self.output_layer.forward(x)
	def backward(self, y, y_pred):
		grads = [self.output_layer.backward(y, y_pred, 1)]
		for layer, x_local in zip(self.layers[::-1], self.x[:-1:][::-1]):
			grad = layer.backward(x_local, grads[-1])
			grads.append(grad)
		return grads[::-1]
	def train(self, x, y, eta, iterations, quiet=False, validate=False, validate_x=None, validate_y=None, decay=0):
		eta_val = eta
		for i in range(iterations):
			y_pred = self.forward(x)
			grad = self.backward(y, y_pred)
			for layer in self.layers:
				layer.update(eta_val)
			self.loss.append(self.output_layer.cost(y, y_pred))
			if validate:
				self.val_loss.append(self.output_layer.cost(validate_y, self.predict(validate_x)))
			if not quiet and not i % 1000:
				print('epoch {}/{} - loss {}'.format(i, iterations, self.loss[-1]), end='')
				if validate:
					print(' - val_loss {}'.format(self.val_loss[-1]), end='')
				print('\n', end='')
			if decay:
				eta_val = eta * np.exp(-decay * i)
	def predict(self, x):
		return self.forward(x)

if __name__ == '__main__':

	import matplotlib.pyplot as plt

	N = 100 # number of points per class
	D = 2 # dimensionality
	K = 3 # number of classes
	X = np.zeros((N*K,D)) # data matrix (each row = single example)
	y = np.zeros(N*K, dtype='uint8') # class labels
	for j in range(K):
		ix = range(N*j,N*(j+1))
		r = np.linspace(0.0,1,N) # radius
		t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
		X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
		y[ix] = j

	net = Network()
	net.add_layer('connected', 2, 100)
	net.add_layer('ReLU', 100, 0)
	net.add_layer('connected', 100, 100)
	net.add_layer('ReLU', 100, 0)
	net.add_layer('connected', 100, 3)
	net.train(X, y, 0.2, 2000)

	# plot the resulting classifier
	h = 0.02
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
						np.arange(y_min, y_max, h))
	x_data = np.c_[xx.ravel(), yy.ravel()]
	Z = net.predict(x_data)
	Z = np.argmax(Z, axis=1)
	Z = Z.reshape(xx.shape)
	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
	plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=plt.cm.Spectral, edgecolors='black')
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.show()
