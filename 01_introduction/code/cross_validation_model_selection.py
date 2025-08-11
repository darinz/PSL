import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge

# Generate data
X = np.random.randn(100, 20)
y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(100)

# Grid search with cross-validation
param_grid = {'alpha': np.logspace(-3, 3, 20)}
ridge = Ridge()
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

print(f"Best alpha: {grid_search.best_params_['alpha']}")
print(f"Best CV score: {-grid_search.best_score_:.4f}")

# Plot CV scores
alphas = param_grid['alpha']
cv_scores = -grid_search.cv_results_['mean_test_score']

plt.figure(figsize=(10, 6))
plt.semilogx(alphas, cv_scores, 'b-', linewidth=2)
plt.axvline(x=grid_search.best_params_['alpha'], color='r', linestyle='--', 
           label=f'Best α = {grid_search.best_params_["alpha"]:.3f}')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Cross-Validation MSE')
plt.title('Cross-Validation for Ridge Regression')
plt.legend()
plt.grid(True)
plt.show()
