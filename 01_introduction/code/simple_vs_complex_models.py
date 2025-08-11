import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, MLPRegressor

# When n < p, use simple models
print("=== Small Dataset (n < p) ===")
n_samples, n_features = 50, 100
X = np.random.randn(n_samples, n_features)
y = np.random.randn(n_samples)

# Linear model with regularization
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
print(f"Ridge R²: {ridge.score(X, y):.4f}")

# Compare with complex model (likely to overfit)
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X, y)
print(f"Random Forest R²: {rf.score(X, y):.4f}")

print("\n=== Large Dataset (n >> p) ===")
# When n >> p, complex models can work well
n_samples, n_features = 10000, 100
X = np.random.randn(n_samples, n_features)
y = np.random.randn(n_samples)

# Complex model with large dataset
mlp = MLPRegressor(hidden_layer_sizes=(200, 100, 50), max_iter=500)
mlp.fit(X, y)
print(f"Neural Network R²: {mlp.score(X, y):.4f}")
