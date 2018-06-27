#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 09:22:28 2017

@author: sashank
"""

import numpy as np

m = 100
X = 5 * np.random.rand(m, 1) - 3
y = 3 * X**2 + 2 * X + np.random.randn(m, 1)

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly_features.fit_transform(X)

print(X[10])
print(X_poly[10])

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

print("Intercept:", lin_reg.intercept_, "Coefficient:", lin_reg.coef_)

X_new = np.array([[-3.24],[-3], [-2.5], [-2], [-1.5], [-1], [0], [1],[1.5], [2], [2.5], [3]])
X_new_poly = poly_features.fit_transform(X_new)
y_predict = lin_reg.predict(X_new_poly)

import matplotlib.pyplot as plt
plt.plot(X, y, "b.", label="Training Data")
plt.plot(X_new, y_predict, "r-", label="Prediction Curve")
plt.show()