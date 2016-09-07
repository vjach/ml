from abc import ABCMeta, abstractmethod
import numpy as np

class Algo(object):
	__metaclass__ = ABCMeta
	
	@abstractmethod
	def train(x, y): pass

	@abstractmethod
	def predict(x, y): pass

class kNN(Algo):
	def __init__(self, X, Y, params):
		self.X = X
		self.Y = Y 
                self.params = params
                self._fill_default_params()

        def _fill_param(self, name, value):
            if name not in self.params.keys():
                self.params[name] = value

        def _fill_default_params(self):
            self._fill_param('k', 3)
            self._fill_param('weights', np.ones(self.X.shape[1]))
            self._fill_param('classes', max(2, len(np.unique(self.Y))))

        @staticmethod
	def _distance(x1, x2):
		return np.linalg.norm((x1 - x2))

        @staticmethod
        def params_span(ranges):
            ret = []

            # TODO:
            for k in range(1, 16):
                ret.append({'k':k})

            return ret

	def train(self):
            pass

	def _predict(self, x):
                w = self.params['weights']
                k = self.params['k']

		wx = x * w	
		nn = np.array([kNN._distance(a * w, wx) for a in self.X])
		nn = self.Y[nn.argsort()]
		nn = nn[:k]
		return np.argmax(np.bincount(nn))

        def predict (self, X, w=[]):
            return np.array([ self._predict(x) for x in X ])

class Perf(object):
    def __init__(self, Y, target_Y):
        self.Y = Y
        self.target_Y = target_Y
        self.classes = range(max(2, len(np.unique(target_Y))))
        l = len(self.classes)
        self.conf_matrix = np.zeros((l, l))
        self._compute_confusion()

    def _compute_confusion(self):
        for i in range(len(self.Y)):
            real_class = self.target_Y[i]
            eval_class = self.Y[i]
            self.conf_matrix[real_class, eval_class] += 1

    def accuracy(self):
        correct_pred = np.array([ self.conf_matrix[cls, cls] for cls in self.classes])
        return np.sum(correct_pred) / len(self.Y)

    def precision(self, cls = 1):
        correct = self.conf_matrix[cls, cls]
        all_classified = np.sum(self.conf_matrix[:, cls])

        if all_classified == 0:
            return 0

        return correct / all_classified
    
    def recall(self, cls = 1):
        correct = self.conf_matrix[cls, cls]
        all_in_class = np.sum(self.conf_matrix[cls, :])

        if all_in_class == 0:
            return 0

        return correct / all_in_class

    def f1_score(self, cls=1, beta=1):
        p = self.precision(cls)
        r = self.recall(cls)
        
        if p == 0 or r == 0:
            return 0

        beta2 = beta ** 2
        return (1 + beta2) * p * r / (beta2 * p + r)

class Tuner(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def evaluate(self, train_X, train_Y, validate_X, validate_Y, cls_class, params):
        cls = cls_class(train_X, train_Y, params)
        cls.train()
        Y = cls.predict(validate_X)
        perf = Perf(Y, validate_Y)
        return {
                'accuracy': perf.accuracy(), 
                'precision':perf.precision(), 
                'recall':perf.recall(), 
                'f1_score':perf.f1_score()}

    def _loocv(self, cls_class, params):
        res_stats = None
        vcount = len(self.X)
        for i in range(vcount):
            train_X = np.delete(self.X, (i), axis = 0)
            train_Y = np.delete(self.Y, (i), axis = 0)
            validate_X = [self.X[i]]
            validate_Y = [self.Y[i]]
            
            res = self.evaluate(train_X, train_Y, validate_X, validate_Y, cls_class, params)
            if not res_stats:
                res_stats = res
            else:
                for key in res_stats.keys():
                    res_stats[key] += res[key]

            for key in res_stats.keys():
                res_stats[key] /= vcount

        return res_stats
   
    def loocv(self, cls_class, ranges):
        pparams = cls_class.params_span(ranges)
        for params in pparams:
            print 'kNN with params:'
            print params
            stats = self._loocv(cls_class, params)
            print stats

