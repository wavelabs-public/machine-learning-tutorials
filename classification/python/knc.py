#Multilabel Classification
#K-neighbours classifier

import numpy as np

from sklearn.datasets import fetch_mldata

#get dataset from 
mnist = fetch_mldata('MNIST original')

X, y = mnist["data"], mnist["target"]

#split the training set and test set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)

y_multilabel = np.c_[y_train_large, y_train_odd]

from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

knn_clf.predict([X[36000]])