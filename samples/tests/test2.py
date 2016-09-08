import numpy as np
import matplotlib.pyplot as plt

import data
import ml

x, y = data.load('data1.txt')

t = ml.Tuner(x, y)
t.loocv(ml.kNN, {})


