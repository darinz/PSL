import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Generate data
X = np.random.randn(100, 5)
y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(100)

# Define models of increasing complexity
models = {
    'Linear': LinearRegression(),
    'Quadratic': Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression())
    ]),
    'Cubic': Pipeline([
        ('poly', PolynomialFeatures(degree=3)),
        ('linear', LinearRegression())
    ])
}

# Evaluate each model
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    results[name] = -scores.mean()

# Find best model
best_model = min(results, key=results.get)
print(f"Best model: {best_model}")
print(f"Best CV MSE: {results[best_model]:.4f}")

# Print all results
for name, mse in results.items():
    print(f"{name}: CV MSE = {mse:.4f}")
