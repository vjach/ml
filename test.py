import numpy as np
import matplotlib.pyplot as plt

import data
import ml

x, y = data.load('ex2data1.txt')
cls = ml.kNN()
cls.train(x, y)

xmap = np.array([[x1, x2] for x1 in np.linspace(20, 100, 100) for x2 in np.linspace(20, 100, 100)])
print xmap[:, 0]
color = ['r', 'b']
color2 = ['black', 'white']
ymap = [cls.predict(s, k=3, w=[.0, 1.]) for s in xmap]
plt.scatter(xmap[:, 0], xmap[:, 1], c = [ color[c] for c in ymap ])
plt.scatter(x[:, 0], x[:, 1], c = [ color2[c] for c in y ])
plt.show()
