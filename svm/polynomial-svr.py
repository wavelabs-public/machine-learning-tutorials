import numpy as np

m = 100
X = 5 * np.random.rand(m, 1) - 3
y = 3 * X**2 + 2 * X + np.random.randn(m, 1)

from sklearn.svm import SVR

svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
svm_poly_reg.fit(X, y)


#print svm_poly_reg.intercept_

X_new = np.array([[-3], [-2.5], [-2], [-1.5], [-1], [-0.5],[0],[0.5], [1],[1.5], [2], [2.5], [3]])
y_predict = svm_poly_reg.predict(X_new)
print (y_predict)

import matplotlib.pyplot as plt
plt.plot(X, y, "b.")
plt.plot(X_new, y_predict, "r--")
plt.show()