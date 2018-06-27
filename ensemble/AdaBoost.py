from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=10,
    algorithm="SAMME.R",
    learning_rate=0.5
)

ada_clf.fit(X, y)

print(ada_clf.predict([[5, 1.5]]))

