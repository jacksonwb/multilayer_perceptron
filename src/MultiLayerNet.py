#! /usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  MultiLayerNet.py                                                            #
#                                                                              #
#  By - jacksonwb                                                              #
#  Created: Tuesday October 2019 11:43:06 am                                   #
#  Modified: Tuesday Oct 2019 12:16:25 pm                                      #
#  Modified By: jacksonwb                                                      #
# ---------------------------------------------------------------------------- #

import numpy as np
import pickle

class ConnectedLayer:
	def __init__(self, input_n, size):
		self.input_n = input_n
		self.size = size
		self.w = 0.01 * np.random.randn(input_n, size)
		self.b = 0.01 * np.ones([1, size])
		self.v = 0
	def forward(self, x):
		return np.dot(x, self.w) + self.b
	def backward(self, x, grad_out):
		dw = np.dot(x.T, grad_out)
		self.dw = dw
		self.db = np.sum(grad_out, axis=0, keepdims=True)
		return np.dot(grad_out, self.w.T)
	def update(self, eta, reg):
		self.dw += reg * self.w
		self.w += -eta * self.dw
		self.b += -eta * self.db
		return reg * np.sum(self.w * self.w)
	def update_nm(self, eta, reg, mu):
		self.dw += reg * self.w
		v_prev = self.v
		self.v = mu * self.v - eta * self.dw
		self.w += -mu * v_prev + (1 + mu) * self.v
		self.b += -eta * self.db
		return reg * np.sum(self.w * self.w)

class ReLU:
	def __init__(self, input_n, size):
		self.size = size
		self.input_n = input_n
	def forward(self, x):
		return(np.maximum(0, x))
	def backward(self, x, grad_out):
		grad_out[x <= 0 ] = 0
		return grad_out
	def update(self, eta, reg):
		pass
	def update_nm(self, eta, reg, mu):
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
	def train(self, x, y, eta, iterations, reg=0, mu=0, quiet=False, validate=False, validate_x=None, validate_y=None, decay=0):
		eta_val = eta
		for i in range(iterations):
			y_pred = self.forward(x)
			grad = self.backward(y, y_pred)

			# update and manage regularization
			reg_loss = []
			for layer in self.layers:
				reg_loss_local = layer.update(eta_val, reg) if not mu else layer.update_nm(eta_val, reg, 0.1)
				if reg_loss_local:
					reg_loss.append(reg_loss_local)
			reg_loss_term = np.sum(reg_loss) / len(reg_loss) if reg else 0
			self.loss.append(self.output_layer.cost(y, y_pred) + reg_loss_term)

			# validation and stats
			if validate:
				self.val_loss.append(self.output_layer.cost(validate_y, self.predict(validate_x)) + reg_loss_term)
			if not quiet and not i % 1000:
				print('epoch {}/{} - loss {}'.format(i, iterations, self.loss[-1]), end='')
				if validate:
					print(' - val_loss {}'.format(self.val_loss[-1]), end='')
				print('\n', end='')

			# learning rate annealing
			if decay:
				eta_val = eta * np.exp(-decay * i)
	def predict(self, x):
		return self.forward(x)
	def save_model(self, model_name='MLN_model.pkl'):
		with open(model_name, mode='wb') as model_file:
			pickle.dump(self.layers, model_file)
	def load_model(self, model_name='MLN_model.pkl'):
		with open(model_name, 'rb') as model_file:
			self.layers = pickle.load(model_file)