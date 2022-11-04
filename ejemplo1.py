import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]
X = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])
Y = [0, 1, 0, 1, 0, 1]

plt.scatter(x, y)
plt.show()

clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, Y)

w = clf.coef_[0]
print(w)

a = -w[0]/w[1]

xx = np.linspace(0, 12)
yy = a*xx-clf.intercept_[0]/w[1]

h = plt.plot(xx, yy, 'k-', label='linea de division')
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.show()

