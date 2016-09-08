import numpy as np
import matplotlib.pyplot as plt

import data
import ml

x, y = data.load('data1.txt')
x2, y2 = data.load('data2.txt')
params = {'k' : 6, 'weights' : [1, 1]}
cls = ml.kNN(x, y, params)
cls.train()
eval_y = cls.predict(x2)
print eval_y
print y2

p = ml.Perf(eval_y, y2)
print('Accuracy: {}'.format(p.accuracy()))
print('Precision: {}'.format(p.precision()))
print('Recall: {}'.format(p.recall()))
