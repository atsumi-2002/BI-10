import xlrd
import openpyxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


def leer_archivo(columnas=['Precio actual', 'Precio final']):
    data = pd.read_excel('Data10.xlsx')
    data = data[:100]
    x = np.array(data[columnas].values)
    y = (data['Estado'].replace('Bajo', 0).replace('Alto', 1))
    return x, y


def proceso():
    x, y = leer_archivo()

    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(x, y)

    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min(x[:, 0]), max(x[:, 0]))
    yy = a * xx - clf.intercept_[0] / w[1]

    h = plt.plot(xx, yy, 'k-', label='linea de division')
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.ylabel('Precio Final')
    plt.xlabel('Precio Actual')
    plt.legend()
    plt.show()


proceso()






