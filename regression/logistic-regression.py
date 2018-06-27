#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:28:52 2017

@author: sashank
"""
import numpy as np

from sklearn import datasets
iris = datasets.load_iris()

print(list(iris.keys()))

X = iris["data"][:, 3:] #petal width(cm)
y = (iris["target"] == 2).astype(np.int) # 1 if Iris-Virginica, else 0

print(y)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X, y)

X_new = np.linspace(0, 3, 100).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)

print(X_new)
print(y_proba)

import matplotlib.pyplot as plt
plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")
plt.show()

log_reg.predict([[1.7], [1.5]])