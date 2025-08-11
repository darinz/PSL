import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg

def demonstrate_lasso_regression_detailed():
    """Demonstrate comprehensive lasso regression implementation with coordinate descent and variable selection"""
    
    # Generate synthetic data with sparse true coefficients
    np.random.seed(42)
    n, p = 100, 20

    # Create design matrix
    X = np.random.randn(n, p)
    # Add some correlation between predictors
    X[:, 1] = 0.3 * X[:, 0] + 0.7 * np.random.randn(n)
    X[:, 2] = 0.2 * X[:, 0] + 0.8 * np.random.randn(n)

    # True coefficients (sparse: only first 5 are non-zero)
    true_beta = np.zeros(p)
    true_beta[:5] = [3, -2, 1.5, -1, 0.8]

    # Generate response
    y = X @ true_beta + 0.5 * np.random.randn(n)

    print("=== LASSO REGRESSION WITH SPARSE COEFFICIENTS ===")
    print(f"Sample size: {n}")
    print(f"Number of predictors: {p}")
    print(f"True non-zero coefficients: {sum(true_beta != 0)}")
    print(f"True coefficients: {true_beta[:5]}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    # Implement coordinate descent for lasso
    def coordinate_descent_lasso(X, y, lambda_val, max_iter=1000, tol=1e-6):
        """Coordinate descent algorithm for lasso regression"""
        n, p = X.shape
        beta = np.zeros(p)
        
        for iteration in range(max_iter):
            beta_old = beta.copy()
            
            for j in range(p):
                # Compute partial residual
                r_j = y - X @ beta + X[:, j] * beta[j]
                
                # Compute univariate OLS
                x_j_norm_sq = np.sum(X[:, j]**2)
                if x_j_norm_sq > 0:
                    beta_ols = np.dot(X[:, j], r_j) / x_j_norm_sq
                    
                    # Apply soft thresholding
                    threshold = lambda_val / (2 * x_j_norm_sq)
                    if abs(beta_ols) <= threshold:
                        beta[j] = 0
                    else:
                        beta[j] = np.sign(beta_ols) * (abs(beta_ols) - threshold)
            
            # Check convergence
            if np.max(np.abs(beta - beta_old)) < tol:
                break
        
        return beta

    # Implement soft thresholding operator
    def soft_threshold(x, threshold):
        """Soft thresholding operator"""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    print(f"\n=== SOFT THRESHOLDING OPERATOR ===")
    print("Demonstrating soft thresholding with different lambda values...")

    # Demonstrate soft thresholding
    x_vals = np.linspace(-3, 3, 100)
    thresholds = [0.5, 1.0, 1.5]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    for threshold in thresholds:
        y_vals = soft_threshold(x_vals, threshold)
        plt.plot(x_vals, y_vals, label=f'λ = {threshold}')
    plt.plot(x_vals, x_vals, 'k--', alpha=0.5, label='Identity')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title('Soft Thresholding Operator')
    plt.legend()
    plt.grid(True)

    # Lasso with different lambda values
    lambda_values = np.logspace(-3, 1, 50)
    lasso_coefs = []
    lasso_preds = []
    lasso_nonzero = []

    print(f"\n=== LASSO COEFFICIENT PATHS ===")
    print("Computing lasso solutions for different lambda values...")

    for alpha in lambda_values:
        # Fit lasso using sklearn
        lasso = Lasso(alpha=alpha, max_iter=2000)
        lasso.fit(X_train_scaled, y_train_scaled)
        
        # Store results
        lasso_coefs.append(lasso.coef_)
        y_pred = lasso.predict(X_test_scaled)
        lasso_preds.append(y_pred)
        lasso_nonzero.append(np.sum(lasso.coef_ != 0))

    lasso_coefs = np.array(lasso_coefs)
    lasso_preds = np.array(lasso_preds)

    # Compare with coordinate descent
    lambda_test = 0.1
    lasso_sklearn = Lasso(alpha=lambda_test, max_iter=2000)
    lasso_sklearn.fit(X_train_scaled, y_train_scaled)

    lasso_cd = coordinate_descent_lasso(X_train_scaled, y_train_scaled, lambda_test)

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Coefficient paths
    axes[0, 0].semilogx(lambda_values, lasso_coefs)
    axes[0, 0].set_xlabel('Regularization Parameter (λ)')
    axes[0, 0].set_ylabel('Coefficient Values')
    axes[0, 0].set_title('Lasso: Coefficient Paths')
    axes[0, 0].grid(True)

    # Number of non-zero coefficients
    axes[0, 1].semilogx(lambda_values, lasso_nonzero, 'r-')
    axes[0, 1].set_xlabel('Regularization Parameter (λ)')
    axes[0, 1].set_ylabel('Number of Non-zero Coefficients')
    axes[0, 1].set_title('Lasso: Sparsity')
    axes[0, 1].grid(True)

    # Cross-validation for optimal lambda
    cv_scores = []
    for alpha in lambda_values:
        lasso = Lasso(alpha=alpha, max_iter=2000)
        scores = cross_val_score(lasso, X_train_scaled, y_train_scaled, cv=5, scoring='neg_mean_squared_error')
        cv_scores.append(-scores.mean())

    best_idx = np.argmin(cv_scores)
    best_lambda = lambda_values[best_idx]

    axes[0, 2].semilogx(lambda_values, cv_scores, 'g-')
    axes[0, 2].axvline(best_lambda, color='red', linestyle='--', label=f'Best λ = {best_lambda:.4f}')
    axes[0, 2].set_xlabel('Regularization Parameter (λ)')
    axes[0, 2].set_ylabel('Cross-Validation MSE')
    axes[0, 2].set_title('Cross-Validation Performance')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # Compare OLS vs Lasso coefficients
    ols_coefs = linalg.lstsq(X_train_scaled, y_train_scaled, rcond=None)[0]
    best_lasso = Lasso(alpha=best_lambda, max_iter=2000)
    best_lasso.fit(X_train_scaled, y_train_scaled)

    x_pos = np.arange(len(ols_coefs))
    width = 0.35

    axes[1, 0].bar(x_pos - width/2, ols_coefs, width, label='OLS', alpha=0.7)
    axes[1, 0].bar(x_pos + width/2, best_lasso.coef_, width, label=f'Lasso (λ={best_lambda:.4f})', alpha=0.7)
    axes[1, 0].set_xlabel('Predictor Index')
    axes[1, 0].set_ylabel('Coefficient Value')
    axes[1, 0].set_title('OLS vs Lasso Coefficients')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Compare sklearn vs coordinate descent
    axes[1, 1].scatter(lasso_sklearn.coef_, lasso_cd, alpha=0.7)
    axes[1, 1].plot([lasso_sklearn.coef_.min(), lasso_sklearn.coef_.max()], 
                    [lasso_sklearn.coef_.min(), lasso_sklearn.coef_.max()], 'r--')
    axes[1, 1].set_xlabel('Sklearn Lasso Coefficients')
    axes[1, 1].set_ylabel('Coordinate Descent Coefficients')
    axes[1, 1].set_title('Implementation Comparison')
    axes[1, 1].grid(True)

    # Prediction comparison
    y_test_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
    ols_pred = scaler_y.inverse_transform((X_test_scaled @ ols_coefs).reshape(-1, 1)).ravel()
    lasso_pred = scaler_y.inverse_transform((X_test_scaled @ best_lasso.coef_).reshape(-1, 1)).ravel()

    axes[1, 2].scatter(y_test_original, ols_pred, alpha=0.6, label=f'OLS (R²={r2_score(y_test_original, ols_pred):.3f})')
    axes[1, 2].scatter(y_test_original, lasso_pred, alpha=0.6, label=f'Lasso (R²={r2_score(y_test_original, lasso_pred):.3f})')
    axes[1, 2].plot([y_test_original.min(), y_test_original.max()], 
                    [y_test_original.min(), y_test_original.max()], 'k--', alpha=0.5)
    axes[1, 2].set_xlabel('True Values')
    axes[1, 2].set_ylabel('Predicted Values')
    axes[1, 2].set_title('Prediction Comparison')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.show()

    # Print results
    print(f"\n=== MODEL RESULTS ===")
    print(f"Best Lasso λ: {best_lambda:.4f}")
    print(f"Non-zero coefficients: {np.sum(best_lasso.coef_ != 0)}")
    print(f"OLS Test R²: {r2_score(y_test_original, ols_pred):.4f}")
    print(f"Lasso Test R²: {r2_score(y_test_original, lasso_pred):.4f}")
    print(f"OLS Test MSE: {mean_squared_error(y_test_original, ols_pred):.4f}")
    print(f"Lasso Test MSE: {mean_squared_error(y_test_original, lasso_pred):.4f}")

    # Compare implementations
    print(f"\n=== IMPLEMENTATION COMPARISON ===")
    print(f"Implementation comparison (λ={lambda_test}):")
    print("Sklearn Lasso:", lasso_sklearn.coef_[:5])
    print("Coordinate Descent:", lasso_cd[:5])
    print("Maximum difference:", np.max(np.abs(lasso_sklearn.coef_ - lasso_cd)))

    # Demonstrate variable selection
    print(f"\n=== VARIABLE SELECTION ANALYSIS ===")
    print("Variable selection results:")
    print("True non-zero coefficients:", np.sum(true_beta != 0))
    print("Lasso non-zero coefficients:", np.sum(best_lasso.coef_ != 0))
    print("Correctly identified non-zero:", np.sum((true_beta != 0) & (best_lasso.coef_ != 0)))
    print("Correctly identified zero:", np.sum((true_beta == 0) & (best_lasso.coef_ == 0)))
    
    # Calculate selection accuracy
    selection_accuracy = (np.sum((true_beta != 0) & (best_lasso.coef_ != 0)) + 
                         np.sum((true_beta == 0) & (best_lasso.coef_ == 0))) / p
    print(f"Variable selection accuracy: {selection_accuracy:.4f}")

    # Additional analysis: Coefficient stability
    print(f"\n=== COEFFICIENT STABILITY ===")
    # Test with different lambda values
    lambda_stability = [0.05, 0.1, 0.2]
    stability_results = []
    
    for lam in lambda_stability:
        lasso_stable = Lasso(alpha=lam, max_iter=2000)
        lasso_stable.fit(X_train_scaled, y_train_scaled)
        stability_results.append(lasso_stable.coef_)
    
    stability_results = np.array(stability_results)
    coefficient_variance = np.var(stability_results, axis=0)
    
    print(f"Coefficient variance across lambda values: {np.mean(coefficient_variance):.6f}")
    print(f"Most stable coefficients (lowest variance): {np.argsort(coefficient_variance)[:5]}")
    print(f"Least stable coefficients (highest variance): {np.argsort(coefficient_variance)[-5:]}")

    # Key insights
    print(f"\n=== KEY INSIGHTS ===")
    print("1. Lasso performs automatic variable selection through soft thresholding")
    print("2. Coordinate descent provides an efficient algorithm for lasso optimization")
    print("3. Cross-validation helps select optimal regularization parameter")
    print("4. Lasso can improve prediction accuracy in sparse settings")
    print("5. Variable selection accuracy depends on signal strength and noise level")
    print("6. Coefficient stability varies across different lambda values")
    print("7. Lasso provides interpretable models through sparsity")
    
    return {
        'best_lambda': best_lambda,
        'lasso_coefs': best_lasso.coef_,
        'ols_coefs': ols_coefs,
        'lasso_r2': r2_score(y_test_original, lasso_pred),
        'ols_r2': r2_score(y_test_original, ols_pred),
        'lasso_mse': mean_squared_error(y_test_original, lasso_pred),
        'ols_mse': mean_squared_error(y_test_original, ols_pred),
        'non_zero_count': np.sum(best_lasso.coef_ != 0),
        'selection_accuracy': selection_accuracy,
        'true_beta': true_beta,
        'lambda_values': lambda_values,
        'lasso_coefs_path': lasso_coefs
    }

# Run demonstration
results = demonstrate_lasso_regression_detailed()
