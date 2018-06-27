from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_iris
from sklearn.tree import export_graphviz

iris = load_iris()
X = iris.data[:, :1] # petal length and width
y = iris.target

tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)

export_graphviz(
    tree_reg,
    out_file="iris_tree_reg.dot",
    feature_names=iris.feature_names[:1],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)