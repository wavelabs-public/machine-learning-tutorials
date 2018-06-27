from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')

X, y = mnist["data"], mnist["target"]

#train test sets split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(max_depth=4),
    n_estimators=10,
    max_samples=100,
    bootstrap=True
)

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
print(bag_clf.score(X_test, y_test))