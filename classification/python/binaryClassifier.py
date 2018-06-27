from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')

X, y = mnist["data"], mnist["target"]

#train test sets split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

import numpy as np

y_train_5 = (y_train == 5)
y_test_5 = (y_train == 5)

#SGD Classifier
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([X[36000]])

from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=10, scoring="accuracy")

from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)

print (y_train_pred)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)

from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred, average='micro')
recall_score(y_train_5, y_train_pred, average='micro')

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

import matplotlib.pyplot as plt

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
	plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
	plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
	plt.xlabel("Threshold")
	plt.legend(loc="upper left")
	plt.ylim([0, 1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()


from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
