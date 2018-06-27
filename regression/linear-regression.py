#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 09:02:44 2017

@author: sashank
"""

import numpy as np

m = 100
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)

print("Intercept:", lin_reg.intercept_, "Coefficient:", lin_reg.coef_)

X_new = np.array([[0], [3]])
y_predict = lin_reg.predict(X_new)

print(X_new)
print( y_predict)

import matplotlib.pyplot as plt

plt.plot(X, y, "b.", label="Training Data")
plt.plot(X_new, y_predict, "r-", label="Prediction")
plt.show()