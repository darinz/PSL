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

def linear_regression_analysis(X, y):
    """
    Complete linear regression analysis
    
    Parameters:
    X: design matrix (n x (p+1)) including intercept column
    y: response vector (n x 1)
    
    Returns:
    Dictionary containing all regression results
    """
    n, p_plus_1 = X.shape
    p = p_plus_1 - 1  # Number of predictors (excluding intercept)
    
    # Compute least squares estimates
    beta_hat = least_squares_estimation(X, y)
    
    # Compute fitted values
    y_hat = X @ beta_hat
    
    # Compute residuals
    residuals = y - y_hat
    
    # Compute RSS and error variance
    RSS = np.sum(residuals**2)
    sigma2_hat = RSS / (n - p - 1)
    
    # Compute coefficient standard errors
    XtX_inv = np.linalg.inv(X.T @ X)
    se_beta = np.sqrt(sigma2_hat * np.diag(XtX_inv))
    
    # Compute t-statistics and p-values
    t_stats = beta_hat / se_beta
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
    
    # Compute R-squared
    y_mean = np.mean(y)
    TSS = np.sum((y - y_mean)**2)  # Total sum of squares
    R_squared = 1 - RSS / TSS
    
    # Compute adjusted R-squared
    R_squared_adj = 1 - (RSS / (n - p - 1)) / (TSS / (n - 1))
    
    return {
        'coefficients': beta_hat,
        'standard_errors': se_beta,
        't_statistics': t_stats,
        'p_values': p_values,
        'fitted_values': y_hat,
        'residuals': residuals,
        'RSS': RSS,
        'sigma2_hat': sigma2_hat,
        'R_squared': R_squared,
        'R_squared_adj': R_squared_adj,
        'degrees_of_freedom': n - p - 1
    }

# Example usage
def create_example_data(n=100, p=2, noise_std=0.5):
    """Create synthetic data for demonstration"""
    np.random.seed(42)
    X_raw = np.random.randn(n, p)
    X = np.column_stack([np.ones(n), X_raw])
    beta_true = np.array([1.0, 2.0, -1.5])
    y = X @ beta_true + noise_std * np.random.randn(n)
    return X, y, beta_true

# Run complete analysis
X, y, beta_true = create_example_data()
results = linear_regression_analysis(X, y)

print("=== Linear Regression Results ===")
print(f"Coefficients: {results['coefficients']}")
print(f"Standard Errors: {results['standard_errors']}")
print(f"R-squared: {results['R_squared']:.4f}")
print(f"Adjusted R-squared: {results['R_squared_adj']:.4f}")
print(f"Error variance estimate: {results['sigma2_hat']:.4f}")
