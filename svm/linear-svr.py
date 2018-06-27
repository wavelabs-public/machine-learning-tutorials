import numpy as np

m = 100
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)


from sklearn.svm import LinearSVR

svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X, y)

svm_reg2 = LinearSVR(epsilon=0.5)
svm_reg2.fit(X, y)

print("Predict:",svm_reg.predict([[1.5]]))

print "Intercept",svm_reg.intercept_
print "Coefficient:",svm_reg.coef_

X_line1 = np.array([[0], [2]])
y_line1 = svm_reg.intercept_ + X_line1 * svm_reg.coef_

X_line2 = np.array([[0], [2]])
y_line2 = svm_reg2.intercept_ + X_line2 * svm_reg2.coef_

import matplotlib.pyplot as plt

plt.plot(X, y, "b.", label="Training Data")
plt.plot(X_line1, y_line1, "g-")
plt.plot(X_line2, y_line2, "r-")
plt.xlabel("x1")
plt.ylabel("y")
plt.show()