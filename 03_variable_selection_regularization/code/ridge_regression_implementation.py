import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg

def demonstrate_ridge_regression():
    """Demonstrate ridge regression implementation with multicollinearity handling"""
    
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

    best_alpha_idx = np.argmin(cv_scores)
    best_alpha = lambda_values[best_alpha_idx]

    axes[1, 0].semilogx(lambda_values, cv_scores, 'r-')
    axes[1, 0].axvline(x=best_alpha, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Regularization Parameter (λ)')
    axes[1, 0].set_ylabel('Cross-Validation MSE')
    axes[1, 0].set_title(f'Cross-Validation (Best λ = {best_alpha:.4f})')
    axes[1, 0].grid(True)

    # Test performance comparison
    test_mses = []
    for i, alpha in enumerate(lambda_values):
        mse = mean_squared_error(y_test, ridge_preds[i])
        test_mses.append(mse)

    axes[1, 1].semilogx(lambda_values, test_mses, 'g-')
    axes[1, 1].axvline(x=best_alpha, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Regularization Parameter (λ)')
    axes[1, 1].set_ylabel('Test MSE')
    axes[1, 1].set_title('Test Performance')
    axes[1, 1].grid(True)

    # Correlation matrix heatmap
    corr_matrix = np.corrcoef(X_train_scaled.T)
    im = axes[1, 2].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 2].set_title('Correlation Matrix')
    axes[1, 2].set_xlabel('Variable Index')
    axes[1, 2].set_ylabel('Variable Index')
    plt.colorbar(im, ax=axes[1, 2])

    plt.tight_layout()
    plt.show()

    # Final model with optimal lambda
    best_ridge = Ridge(alpha=best_alpha)
    best_ridge.fit(X_train_scaled, y_train_scaled)

    # Transform coefficients back to original scale
    ridge_coef_original = best_ridge.coef_ * scaler_y.scale_ / scaler_X.scale_
    ridge_intercept = scaler_y.mean_ - np.sum(ridge_coef_original * scaler_X.mean_)

    # Model evaluation
    y_pred_final = ridge_intercept + X_test @ ridge_coef_original
    final_mse = mean_squared_error(y_test, y_pred_final)
    final_r2 = r2_score(y_test, y_pred_final)

    print("=== RIDGE REGRESSION RESULTS ===")
    print(f"Best regularization parameter (λ): {best_alpha:.4f}")
    print(f"Test MSE: {final_mse:.4f}")
    print(f"Test R²: {final_r2:.4f}")
    print(f"Degrees of freedom: {ridge_dfs[best_alpha_idx]:.2f}")
    
    print("\nCoefficients (original scale):")
    for i, coef in enumerate(ridge_coef_original):
        print(f"  β_{i}: {coef:.4f}")
    
    print(f"\nIntercept: {ridge_intercept:.4f}")

    # Compare with OLS
    from sklearn.linear_model import LinearRegression
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train_scaled)
    ols_coef_original = ols.coef_ * scaler_y.scale_ / scaler_X.scale_
    ols_intercept = scaler_y.mean_ - np.sum(ols_coef_original * scaler_X.mean_)
    
    ols_pred = ols_intercept + X_test @ ols_coef_original
    ols_mse = mean_squared_error(y_test, ols_pred)
    ols_r2 = r2_score(y_test, ols_pred)

    print("\n=== COMPARISON WITH OLS ===")
    print(f"OLS Test MSE: {ols_mse:.4f}")
    print(f"OLS Test R²: {ols_r2:.4f}")
    print(f"Ridge Test MSE: {final_mse:.4f}")
    print(f"Ridge Test R²: {final_r2:.4f}")
    print(f"Improvement in MSE: {((ols_mse - final_mse) / ols_mse * 100):.2f}%")

    # Key insights
    print("\n=== KEY INSIGHTS ===")
    print("1. Ridge regression handles multicollinearity effectively")
    print("2. Cross-validation helps select optimal regularization parameter")
    print("3. Ridge shrinks coefficients but doesn't set them to zero")
    print("4. Degrees of freedom decrease with increasing regularization")
    print("5. Ridge can improve prediction accuracy in presence of multicollinearity")
    
    return best_ridge, ridge_coef_original, ridge_intercept, lambda_values, cv_scores

# Run demonstration
best_ridge, ridge_coef_original, ridge_intercept, lambda_values, cv_scores = demonstrate_ridge_regression()
