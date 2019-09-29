#! /usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  MultiLayerNet.py                                                            #
#                                                                              #
#  By - jacksonwb                                                              #
#  Created: Wednesday December 1969 4:00:00 pm                                 #
#  Modified: Saturday Sep 2019 5:09:24 pm                                      #
#  Modified By: jacksonwb                                                      #
# ---------------------------------------------------------------------------- #

import numpy as np

class ConnectedLayer:
	def __init__(self, input_n, size):
		self.input_n = input_n
		self.size = size
		self.w = 0.01 * np.random.randn(size, input_n)
		self.b = 0.01 * np.ones([1, size])
	def forward(self, x):
		return np.dot(x, self.w) + self.b
	def backward(self, x, grad_out):
		dw = np.dot(self.x.T, grad_out)
		self.dw = dw
		self.db = np.sum(grad_out, axis=0, keepdims=True)
		return dw
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
		return exp / np.sum(exp)
	def backward(self, y, y_pred, grad_out):
		p = y_pred
		p[range(y.shape[0]), y] -= 1
		p /= y.shape[0]
		return p * grad_out
	def cost(self, y, y_pred):
		n = y.shape[0]
		log_likelihood = -np.log(y_pred[range(n),y])
		loss = np.sum(log_likelihood) / n
		return loss

layer_types = {'ReLU':ReLU, 'connected':ConnectedLayer}

class Network:
	def __init__(self, input_n, output_n):
		self.input_n = input_n
		self.output_n = output_n
		self.trained = False
		self.layers = []
		self.output_layer = SoftmaxCrossEntropy()
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
		for layer, x in zip(self.layers[::-1], self.x[::-1]):
			grad = layer.backward(x, grad)
			grads.append(grad)
		return grads[::-1]
	def train(self, x, y, eta, iterations, quiet=False):
		for i in range(iterations):
			y_pred = self.forward(x)
			grad = self.backward(y, y_pred)
			for layer in self.layers:
				layer.update(eta)
			if not quiet:
				print('epoch {}/{} - loss {}'.format(i, iterations, self.output_layer.cost(y, y_pred)))
	def predict(self, x):
		return self.forward(x)


