import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def demonstrate_regularization_comparison():
    """Demonstrate ridge vs lasso regularization comparison"""
    
    # Generate synthetic data
    np.random.seed(42)
    n, p = 100, 20
    X = np.random.randn(n, p)
    true_beta = np.zeros(p)
    true_beta[:5] = [2, -1.5, 1, -0.8, 0.6]  # Only first 5 coefficients are non-zero
    y = X @ true_beta + 0.5 * np.random.randn(n)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    # Fit models
    lambda_values = np.logspace(-3, 3, 50)

    # Ridge regression
    ridge_scores = []
    ridge_coefs = []

    for alpha in lambda_values:
        ridge = Ridge(alpha=alpha)
        scores = cross_val_score(ridge, X_train_scaled, y_train_scaled, cv=5, scoring='neg_mean_squared_error')
        ridge_scores.append(-scores.mean())
        
        ridge.fit(X_train_scaled, y_train_scaled)
        ridge_coefs.append(ridge.coef_)

    # Lasso regression
    lasso_scores = []
    lasso_coefs = []

    for alpha in lambda_values:
        lasso = Lasso(alpha=alpha, max_iter=2000)
        scores = cross_val_score(lasso, X_train_scaled, y_train_scaled, cv=5, scoring='neg_mean_squared_error')
        lasso_scores.append(-scores.mean())
        
        lasso.fit(X_train_scaled, y_train_scaled)
        lasso_coefs.append(lasso.coef_)

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Cross-validation scores
    axes[0, 0].semilogx(lambda_values, ridge_scores, 'b-', label='Ridge')
    axes[0, 0].semilogx(lambda_values, lasso_scores, 'r-', label='Lasso')
    axes[0, 0].set_xlabel('Regularization Parameter (λ)')
    axes[0, 0].set_ylabel('Cross-Validation MSE')
    axes[0, 0].set_title('Cross-Validation Performance')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Coefficient paths
    ridge_coefs = np.array(ridge_coefs)
    lasso_coefs = np.array(lasso_coefs)

    axes[0, 1].semilogx(lambda_values, ridge_coefs)
    axes[0, 1].set_xlabel('Regularization Parameter (λ)')
    axes[0, 1].set_ylabel('Coefficient Values')
    axes[0, 1].set_title('Ridge: Coefficient Paths')
    axes[0, 1].grid(True)

    axes[1, 0].semilogx(lambda_values, lasso_coefs)
    axes[1, 0].set_xlabel('Regularization Parameter (λ)')
    axes[1, 0].set_ylabel('Coefficient Values')
    axes[1, 0].set_title('Lasso: Coefficient Paths')
    axes[1, 0].grid(True)

    # Sparsity comparison
    ridge_nonzero = np.sum(ridge_coefs != 0, axis=1)
    lasso_nonzero = np.sum(lasso_coefs != 0, axis=1)

    axes[1, 1].semilogx(lambda_values, ridge_nonzero, 'b-', label='Ridge')
    axes[1, 1].semilogx(lambda_values, lasso_nonzero, 'r-', label='Lasso')
    axes[1, 1].set_xlabel('Regularization Parameter (λ)')
    axes[1, 1].set_ylabel('Number of Non-zero Coefficients')
    axes[1, 1].set_title('Sparsity Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    # Optimal lambda selection
    best_ridge_idx = np.argmin(ridge_scores)
    best_lasso_idx = np.argmin(lasso_scores)

    print(f"Best Ridge λ: {lambda_values[best_ridge_idx]:.4f}")
    print(f"Best Lasso λ: {lambda_values[best_lasso_idx]:.4f}")

    # Final model evaluation
    best_ridge = Ridge(alpha=lambda_values[best_ridge_idx])
    best_lasso = Lasso(alpha=lambda_values[best_lasso_idx])

    best_ridge.fit(X_train_scaled, y_train_scaled)
    best_lasso.fit(X_train_scaled, y_train_scaled)

    # Transform coefficients back to original scale
    ridge_coef_original = best_ridge.coef_ * scaler_y.scale_ / scaler_X.scale_
    lasso_coef_original = best_lasso.coef_ * scaler_y.scale_ / scaler_X.scale_

    ridge_intercept = scaler_y.mean_ - np.sum(ridge_coef_original * scaler_X.mean_)
    lasso_intercept = scaler_y.mean_ - np.sum(lasso_coef_original * scaler_X.mean_)

    print("\nRidge Regression Results:")
    print(f"Intercept: {ridge_intercept:.4f}")
    print(f"Non-zero coefficients: {np.sum(ridge_coef_original != 0)}")
    print(f"Test R²: {r2_score(y_test, ridge_intercept + X_test @ ridge_coef_original):.4f}")

    print("\nLasso Regression Results:")
    print(f"Intercept: {lasso_intercept:.4f}")
    print(f"Non-zero coefficients: {np.sum(lasso_coef_original != 0)}")
    print(f"Test R²: {r2_score(y_test, lasso_intercept + X_test @ lasso_coef_original):.4f}")
    
    return ridge_coef_original, lasso_coef_original, lambda_values, ridge_scores, lasso_scores

# Run demonstration
ridge_coef_original, lasso_coef_original, lambda_values, ridge_scores, lasso_scores = demonstrate_regularization_comparison()
