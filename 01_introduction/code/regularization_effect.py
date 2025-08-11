import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split

# Generate data
X = np.random.randn(100, 20)
y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Test different regularization strengths
alphas = np.logspace(-3, 3, 20)
ridge_scores = []
lasso_scores = []

for alpha in alphas:
    # Ridge regression
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    ridge_scores.append(ridge.score(X_test, y_test))
    
    # Lasso regression
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    lasso_scores.append(lasso.score(X_test, y_test))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.semilogx(alphas, ridge_scores, 'b-', label='Ridge')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('R² Score')
plt.title('Ridge Regression')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.semilogx(alphas, lasso_scores, 'r-', label='Lasso')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('R² Score')
plt.title('Lasso Regression')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
