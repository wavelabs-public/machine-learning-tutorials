import numpy as np

m = 100
X = 5 * np.random.rand(m, 1) - 3
y = 3 * X**2 + 2 * X + np.random.randn(m, 1)
y = y.reshape(y.shape[0])

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import GradientBoostingRegressor

grbt = GradientBoostingRegressor(max_depth=3, n_estimators=5, learning_rate=1.0)
grbt.fit(X, y)

X_test = np.arange(-3, 2, 0.1)
X_test = X_test.reshape(X_test.size, 1)
y_pred = grbt.predict(X_test)

import matplotlib.pyplot as plt

plt.plot(X, y, "b.", label="Training Data")
plt.plot(X_test, y_pred, "r-", label="Predicted Data")
plt.show()