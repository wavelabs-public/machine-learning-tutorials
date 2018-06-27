import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]
y = (iris["target"] == 2).astype(np.float64)
'''
svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear", LinearSVC(C = 1, loss="hinge")),
])
'''
svm_clf = LinearSVC(C = 1, loss="hinge")

svm_clf.fit(X, y)

print("Intercept:", svm_clf.intercept_)
print("Coef:", svm_clf.coef_)

print(svm_clf.predict([[5.5, 1.7]]))

import matplotlib.pyplot as plt


plt.plot(X[:, 0], X[:, 1], "b.")
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.show()

