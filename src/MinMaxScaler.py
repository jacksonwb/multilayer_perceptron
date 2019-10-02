#! /usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  StandardScaler.py                                                           #
#                                                                              #
#  By - jacksonwb                                                              #
#  Created: Wednesday December 1969 4:00:00 pm                                 #
#  Modified: Monday Sep 2019 2:37:43 pm                                        #
#  Modified By: jacksonwb                                                      #
# ---------------------------------------------------------------------------- #

import numpy as np

class MinMaxScaler:
	def __init__(self):
		self.fitted = False

	def fit(self, x):
		self.min = x.min(axis=0)
		self.max = x.max(axis=0)
		self.fitted = True

	def transform(self, x):
		if not self.fitted:
			raise Exception('Scaler has not been fitted')
		return (x - self.min) / (self.max - self.min)

	def save(self):
		return (self.min, self.max)

	def load(self, model):
		self.min = model[0]
		self.max = model[1]
		self.fitted = True


