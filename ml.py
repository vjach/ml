from abc import ABCMeta, abstractmethod
import numpy as np

class Algo(object):
	__metaclass__ = ABCMeta
	
	@abstractmethod
	def train(x, y): pass

	@abstractmethod
	def predict(x, y): pass

class kNN(Algo):
	def __init__(self):
		_data = []
		_target = []

	@staticmethod
	def _distance(x1, x2):
		return np.linalg.norm((x1 - x2))

	def train(self, x, y):
		self._data = np.array(x)
		self._target = np.array(y)

	def predict(self, x, k=3, w=[]):
		if not w:
			w = np.ones(self._data.shape[1])
		wx = x * w	
		nn = np.array([kNN._distance(a * w, wx) for a in self._data])
		nn = self._target[nn.argsort()]
		nn = nn[:k]
		return np.argmax(np.bincount(nn))		
