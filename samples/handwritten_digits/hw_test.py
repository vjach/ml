import numpy as np
import matplotlib.pyplot as plt

import idx
import ml

x_file = 'samples/train_x.idx'
y_file = 'samples/train_y.idx'

X = idx.decode(x_file)
Y = idx.decode(y_file)
tun = ml.Tuner(X[:4000,:], Y[:4000], 10)
tun.loocv(ml.kNN, {})


