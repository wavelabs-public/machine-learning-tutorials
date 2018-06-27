#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 15:12:23 2017

@author: sashank
"""
# regularized linear Regression

import numpy as np

m = 100
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)

from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)

X_new = np.array([[0], [3]])
y_predict = ridge_reg.predict(X_new)

print(X_new)
print( y_predict)

import matplotlib.pyplot as plt

plt.plot(X, y, "b.", label="Training Data")
plt.plot(X_new, y_predict, "r-", label="Prediction")
plt.show()