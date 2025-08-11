import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg

def demonstrate_ridge_regression_detailed():
    """Demonstrate comprehensive ridge regression implementation with multicollinearity handling"""
    
    # Generate synthetic data with multicollinearity
    np.random.seed(42)
    n, p = 100, 10

    # Create correlated predictors
    X = np.random.randn(n, p)
    # Add correlation between predictors
    X[:, 1] = 0.8 * X[:, 0] + 0.2 * np.random.randn(n)
    X[:, 2] = 0.7 * X[:, 0] + 0.3 * np.random.randn(n)

    # True coefficients (only first 3 are non-zero)
    true_beta = np.zeros(p)
    true_beta[:3] = [2, -1.5, 1]

    # Generate response
    y = X @ true_beta + 0.5 * np.random.randn(n)

    print("=== RIDGE REGRESSION WITH MULTICOLLINEARITY ===")
    print(f"Sample size: {n}")
    print(f"Number of predictors: {p}")
    print(f"True non-zero coefficients: {sum(true_beta != 0)}")
    print(f"True coefficients: {true_beta[:3]}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    # Compute SVD
    U, d, Vt = linalg.svd(X_train_scaled, full_matrices=False)

    print(f"\nSVD Analysis:")
    print(f"Singular values: {d[:5]}...")
    print(f"Condition number: {d[0] / d[-1]:.2f}")

    # Ridge regression with different lambda values
    lambda_values = np.logspace(-3, 3, 50)
    ridge_coefs = []
    ridge_preds = []
    ridge_dfs = []

    for alpha in lambda_values:
        # Fit ridge regression
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_scaled, y_train_scaled)
        
        # Store coefficients
        ridge_coefs.append(ridge.coef_)
        
        # Store predictions
        y_pred = ridge.predict(X_test_scaled)
        ridge_preds.append(y_pred)
        
        # Compute degrees of freedom
        df = np.sum(d**2 / (d**2 + alpha))
        ridge_dfs.append(df)

    ridge_coefs = np.array(ridge_coefs)
    ridge_preds = np.array(ridge_preds)

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Coefficient paths
    axes[0, 0].semilogx(lambda_values, ridge_coefs)
    axes[0, 0].set_xlabel('Regularization Parameter (λ)')
    axes[0, 0].set_ylabel('Coefficient Values')
    axes[0, 0].set_title('Ridge: Coefficient Paths')
    axes[0, 0].grid(True)

    # Degrees of freedom
    axes[0, 1].semilogx(lambda_values, ridge_dfs, 'b-')
    axes[0, 1].set_xlabel('Regularization Parameter (λ)')
    axes[0, 1].set_ylabel('Degrees of Freedom')
    axes[0, 1].set_title('Ridge: Degrees of Freedom')
    axes[0, 1].grid(True)

    # Shrinkage factors for different singular values
    shrinkage_factors = d**2 / (d**2[:, None] + lambda_values)
    for i in range(min(5, len(d))):
        axes[0, 2].semilogx(lambda_values, shrinkage_factors[i], label=f'd_{i+1}={d[i]:.2f}')
    axes[0, 2].set_xlabel('Regularization Parameter (λ)')
    axes[0, 2].set_ylabel('Shrinkage Factor')
    axes[0, 2].set_title('Shrinkage Factors by Singular Value')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # Cross-validation for optimal lambda
    cv_scores = []
    for alpha in lambda_values:
        ridge = Ridge(alpha=alpha)
        scores = cross_val_score(ridge, X_train_scaled, y_train_scaled, cv=5, scoring='neg_mean_squared_error')
        cv_scores.append(-scores.mean())

    best_idx = np.argmin(cv_scores)
    best_lambda = lambda_values[best_idx]

    axes[1, 0].semilogx(lambda_values, cv_scores, 'r-')
    axes[1, 0].axvline(best_lambda, color='red', linestyle='--', label=f'Best λ = {best_lambda:.4f}')
    axes[1, 0].set_xlabel('Regularization Parameter (λ)')
    axes[1, 0].set_ylabel('Cross-Validation MSE')
    axes[1, 0].set_title('Cross-Validation Performance')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Compare OLS vs Ridge coefficients
    ols_coefs = linalg.lstsq(X_train_scaled, y_train_scaled, rcond=None)[0]
    best_ridge = Ridge(alpha=best_lambda)
    best_ridge.fit(X_train_scaled, y_train_scaled)

    x_pos = np.arange(len(ols_coefs))
    width = 0.35

    axes[1, 1].bar(x_pos - width/2, ols_coefs, width, label='OLS', alpha=0.7)
    axes[1, 1].bar(x_pos + width/2, best_ridge.coef_, width, label=f'Ridge (λ={best_lambda:.4f})', alpha=0.7)
    axes[1, 1].set_xlabel('Predictor Index')
    axes[1, 1].set_ylabel('Coefficient Value')
    axes[1, 1].set_title('OLS vs Ridge Coefficients')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Prediction comparison
    y_test_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
    ols_pred = scaler_y.inverse_transform((X_test_scaled @ ols_coefs).reshape(-1, 1)).ravel()
    ridge_pred = scaler_y.inverse_transform((X_test_scaled @ best_ridge.coef_).reshape(-1, 1)).ravel()

    axes[1, 2].scatter(y_test_original, ols_pred, alpha=0.6, label=f'OLS (R²={r2_score(y_test_original, ols_pred):.3f})')
    axes[1, 2].scatter(y_test_original, ridge_pred, alpha=0.6, label=f'Ridge (R²={r2_score(y_test_original, ridge_pred):.3f})')
    axes[1, 2].plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'k--', alpha=0.5)
    axes[1, 2].set_xlabel('True Values')
    axes[1, 2].set_ylabel('Predicted Values')
    axes[1, 2].set_title('Prediction Comparison')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.show()

    # Print results
    print(f"\n=== MODEL RESULTS ===")
    print(f"Best Ridge λ: {best_lambda:.4f}")
    print(f"Degrees of Freedom: {ridge_dfs[best_idx]:.2f}")
    print(f"OLS Test R²: {r2_score(y_test_original, ols_pred):.4f}")
    print(f"Ridge Test R²: {r2_score(y_test_original, ridge_pred):.4f}")
    print(f"OLS Test MSE: {mean_squared_error(y_test_original, ols_pred):.4f}")
    print(f"Ridge Test MSE: {mean_squared_error(y_test_original, ridge_pred):.4f}")

    # Demonstrate the augmented data interpretation
    def ridge_via_augmented_data(X, y, lambda_val):
        """Implement ridge regression using the augmented data approach"""
        n, p = X.shape
        
        # Create augmented data
        X_aug = np.vstack([X, np.sqrt(lambda_val) * np.eye(p)])
        y_aug = np.concatenate([y, np.zeros(p)])
        
        # Solve using OLS on augmented data
        beta_aug = linalg.lstsq(X_aug, y_aug, rcond=None)[0]
        
        return beta_aug

    # Compare methods
    lambda_test = 1.0
    ridge_sklearn = Ridge(alpha=lambda_test)
    ridge_sklearn.fit(X_train_scaled, y_train_scaled)

    ridge_augmented = ridge_via_augmented_data(X_train_scaled, y_train_scaled, lambda_test)

    print(f"\n=== AUGMENTED DATA INTERPRETATION ===")
    print(f"Coefficient comparison (λ={lambda_test}):")
    print("Sklearn Ridge:", ridge_sklearn.coef_[:3])
    print("Augmented Data:", ridge_augmented[:3])
    print("Maximum difference:", np.max(np.abs(ridge_sklearn.coef_ - ridge_augmented)))

    # Additional analysis: Multicollinearity assessment
    print(f"\n=== MULTICOLLINEARITY ANALYSIS ===")
    corr_matrix = np.corrcoef(X_train_scaled.T)
    max_corr = np.max(np.abs(corr_matrix - np.eye(p)))
    print(f"Maximum correlation between predictors: {max_corr:.4f}")
    
    # Variance Inflation Factor (VIF) approximation
    vif_values = []
    for i in range(p):
        # Regress predictor i on all other predictors
        X_others = np.delete(X_train_scaled, i, axis=1)
        y_pred = X_train_scaled[:, i]
        try:
            coef_others = linalg.lstsq(X_others, y_pred, rcond=None)[0]
            y_pred_fitted = X_others @ coef_others
            r_squared = 1 - np.sum((y_pred - y_pred_fitted)**2) / np.sum((y_pred - np.mean(y_pred))**2)
            vif = 1 / (1 - r_squared) if r_squared < 1 else np.inf
            vif_values.append(vif)
        except:
            vif_values.append(np.inf)
    
    print(f"VIF values: {[f'{v:.2f}' if v != np.inf else 'inf' for v in vif_values[:5]]}...")
    print(f"Maximum VIF: {max([v for v in vif_values if v != np.inf]):.2f}")

    # Key insights
    print(f"\n=== KEY INSIGHTS ===")
    print("1. Ridge regression handles multicollinearity effectively")
    print("2. Cross-validation helps select optimal regularization parameter")
    print("3. Ridge shrinks coefficients but doesn't set them to zero")
    print("4. Degrees of freedom decrease with increasing regularization")
    print("5. Augmented data interpretation provides geometric insight")
    print("6. Ridge can improve prediction accuracy in presence of multicollinearity")
    
    return {
        'best_lambda': best_lambda,
        'ridge_dfs': ridge_dfs,
        'ols_coefs': ols_coefs,
        'ridge_coefs': best_ridge.coef_,
        'ols_r2': r2_score(y_test_original, ols_pred),
        'ridge_r2': r2_score(y_test_original, ridge_pred),
        'ols_mse': mean_squared_error(y_test_original, ols_pred),
        'ridge_mse': mean_squared_error(y_test_original, ridge_pred),
        'singular_values': d,
        'vif_values': vif_values
    }

# Run demonstration
results = demonstrate_ridge_regression_detailed()
