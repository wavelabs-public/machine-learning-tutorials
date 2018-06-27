#Multiclass Classification
#Random forest classifier

from sklearn.datasets import fetch_mldata

#get dataset from 
mnist = fetch_mldata('MNIST original')

X, y = mnist["data"], mnist["target"]

#split the training set and test set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


import numpy as np

shuffle_index = np.random.permutation(60000)

#shuffle training set
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest_clf.fit(X_train, y_train)
forest_clf.predict([X[36000]])