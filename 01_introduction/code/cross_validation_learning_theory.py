import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

# Generate synthetic data
def generate_data(n_samples=1000, n_features=10):
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(n_samples)
    return X, y

# Cross-validation with different regularization strengths
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
cv_scores = []

X, y = generate_data(n_samples=1000, n_features=10)

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    scores = cross_val_score(ridge, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_scores.append(-scores.mean())  # Convert to positive MSE

# Find best alpha
best_alpha = alphas[np.argmin(cv_scores)]
print(f"Best alpha: {best_alpha}")
print(f"Best CV MSE: {min(cv_scores):.4f}")

# Print all results
for alpha, mse in zip(alphas, cv_scores):
    print(f"Alpha: {alpha}, CV MSE: {mse:.4f}")
