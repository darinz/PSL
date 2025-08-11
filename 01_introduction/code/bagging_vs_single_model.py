import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Generate data
X = np.random.randn(100, 20)
y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Single decision tree
single_tree = DecisionTreeRegressor(max_depth=10)
single_tree.fit(X_train, y_train)
single_score = single_tree.score(X_test, y_test)

# Bagging ensemble
bagging = BaggingRegressor(
    DecisionTreeRegressor(max_depth=10),
    n_estimators=100,
    random_state=42
)
bagging.fit(X_train, y_train)
bagging_score = bagging.score(X_test, y_test)

print(f"Single Tree R²: {single_score:.4f}")
print(f"Bagging R²: {bagging_score:.4f}")
