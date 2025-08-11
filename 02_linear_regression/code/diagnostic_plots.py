import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def least_squares_estimation(X, y):
    """Compute least squares estimates using the normal equation"""
    XtX = X.T @ X
    if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
        print("Warning: X^T X is not full rank. Solution may not be unique.")
    beta_hat = np.linalg.inv(XtX) @ X.T @ y
    return beta_hat

def diagnostic_plots(X, y, results):
    """Create diagnostic plots for linear regression"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Residuals vs Fitted
    axes[0,0].scatter(results['fitted_values'], results['residuals'], alpha=0.6)
    axes[0,0].axhline(y=0, color='red', linestyle='--')
    axes[0,0].set_xlabel('Fitted Values')
    axes[0,0].set_ylabel('Residuals')
    axes[0,0].set_title('Residuals vs Fitted')
    
    # 2. Q-Q Plot
    stats.probplot(results['residuals'], dist="norm", plot=axes[0,1])
    axes[0,1].set_title('Normal Q-Q Plot')
    
    # 3. Scale-Location Plot
    standardized_residuals = results['residuals'] / np.sqrt(results['sigma2_hat'])
    axes[1,0].scatter(results['fitted_values'], np.abs(standardized_residuals), alpha=0.6)
    axes[1,0].set_xlabel('Fitted Values')
    axes[1,0].set_ylabel('|Standardized Residuals|')
    axes[1,0].set_title('Scale-Location Plot')
    
    # 4. Residuals vs Leverage
    # Compute leverage (hat matrix diagonal)
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    leverage = np.diag(H)
    axes[1,1].scatter(leverage, standardized_residuals, alpha=0.6)
    axes[1,1].set_xlabel('Leverage')
    axes[1,1].set_ylabel('Standardized Residuals')
    axes[1,1].set_title('Residuals vs Leverage')
    
    plt.tight_layout()
    plt.show()

# Example usage
def create_example_data(n=100, p=2, noise_std=0.5):
    """Create synthetic data for demonstration"""
    np.random.seed(42)
    X_raw = np.random.randn(n, p)
    X = np.column_stack([np.ones(n), X_raw])
    beta_true = np.array([1.0, 2.0, -1.5])
    y = X @ beta_true + noise_std * np.random.randn(n)
    return X, y, beta_true

def linear_regression_analysis(X, y):
    """Complete linear regression analysis"""
    n, p_plus_1 = X.shape
    p = p_plus_1 - 1
    
    beta_hat = least_squares_estimation(X, y)
    y_hat = X @ beta_hat
    residuals = y - y_hat
    RSS = np.sum(residuals**2)
    sigma2_hat = RSS / (n - p - 1)
    
    return {
        'fitted_values': y_hat,
        'residuals': residuals,
        'sigma2_hat': sigma2_hat
    }

# Generate data and create diagnostic plots
X, y, beta_true = create_example_data()
results = linear_regression_analysis(X, y)
diagnostic_plots(X, y, results)
