#! /usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  MLP.py                                                                      #
#                                                                              #
#  By - jacksonwb                                                              #
#  Created: Tuesday October 2019 11:43:21 am                                   #
#  Modified: Tuesday Oct 2019 3:23:42 pm                                       #
#  Modified By: jacksonwb                                                      #
# ---------------------------------------------------------------------------- #

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import src.MultiLayerNet as MLN
import matplotlib.pyplot as plt
import src.MinMaxScaler as MMS

def parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('data')
	parser.add_argument('-p', '--predict')
	return parser.parse_args()

def get_data(input):
	try:
		data = pd.read_csv(input)
	except FileNotFoundError as e:
		print('MLP:', e)
		exit(1)
	return data

def save_model(mln_model, scaler_model, model_name='MLP_model.pkl'):
	with open(model_name, mode='wb') as model_file:
		pickle.dump((scaler_model, mln_model), model_file)

def load_model(model_name='MLP_model.pkl'):
	with open(model_name, 'rb') as model_file:
		return pickle.load(model_file)

if __name__ == '__main__':
	args = parse()
	data = get_data(args.data)
	data.columns = ['id', 'diagnose', 'radius_m', 'texture_m', 'perimeter_m', 'area_m',
		'smoothness_m', 'compactness_m', 'concavity_m', 'concave_points_m', 'symmetry_m',
		'fractal_dim_m','radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
		'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se','fractal_dim_se',
		'radius_l', 'texture_l', 'perimeter_l', 'area_l', 'smoothness_l', 'compactness_l',
		'concavity_l', 'concave_points_l', 'symmetry_l','fractal_dim_l']

	# clean data
	X = data.copy()
	y = data['diagnose'].copy()
	del(X['id'], X['diagnose'], X['perimeter_m'], X['area_m'], X['perimeter_se'], X['area_se'], X['perimeter_l'], X['area_l'])
	y[y == 'B'] = 0
	y[y == 'M'] = 1
	print('Number of features:', len(X.columns))

	net = MLN.Network()
	scaler = MMS.MinMaxScaler()

	if args.predict:
		model = load_model()
		net.load(model[0])
		scaler.load(model[1])
		x_pred = scaler.transform(X.to_numpy())
		y_pred = net.predict(x_pred).argmax(axis=1).astype(object)
		y_pred[y_pred == 0] = 'B'
		y_pred[y_pred == 1] = 'M'
		print(y_pred)
	else:
		scaler.fit(X.to_numpy())
		x_scaled = scaler.transform(X.to_numpy())
		net.add_layer('connected', 24, 100)
		net.add_layer('ReLU', 100, 0)
		net.add_layer('connected', 100, 100)
		net.add_layer('ReLU', 100, 0),
		net.add_layer('connected', 100, 100)
		net.add_layer('ReLU', 100, 0),
		net.add_layer('connected', 100, 2)
		x_train, x_test, y_train, y_test = train_test_split(x_scaled, y.to_numpy().astype('int'), test_size=0.2)
		net.train(x_train, y_train, 0.1, 6000, reg=0.001, mu=0.01,
			validate=True, validate_x = x_test, validate_y = y_test,
			decay=0.0001)
		save_model(scaler.save(), net.save())

		y_predict = net.predict(x_test).argmax(axis=1)
		correct = len(y_predict[y_predict == y_test])
		print('Correct: {} / {}'.format(correct, len(y_predict)))
		print('Validation Accuracy:', correct/len(y_predict))

		plt.plot(net.loss)
		plt.plot(net.val_loss)
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['Training Loss', 'Validation Loss'])
		plt.show()

