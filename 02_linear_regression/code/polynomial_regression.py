import numpy as np
import matplotlib.pyplot as plt

def least_squares_estimation(X, y):
    """Compute least squares estimates using the normal equation"""
    XtX = X.T @ X
    if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
        print("Warning: X^T X is not full rank. Solution may not be unique.")
    beta_hat = np.linalg.inv(XtX) @ X.T @ y
    return beta_hat

def linear_regression_analysis(X, y):
    """Complete linear regression analysis"""
    n, p_plus_1 = X.shape
    p = p_plus_1 - 1
    
    beta_hat = least_squares_estimation(X, y)
    y_hat = X @ beta_hat
    residuals = y - y_hat
    RSS = np.sum(residuals**2)
    
    # Compute R-squared
    y_mean = np.mean(y)
    TSS = np.sum((y - y_mean)**2)
    R_squared = 1 - RSS / TSS
    
    return {
        'coefficients': beta_hat,
        'fitted_values': y_hat,
        'residuals': residuals,
        'RSS': RSS,
        'R_squared': R_squared
    }

def polynomial_regression(X, y, degree=2):
    """
    Fit polynomial regression
    
    Parameters:
    X: predictor matrix (single predictor)
    y: response vector
    degree: polynomial degree
    
    Returns:
    Polynomial regression results
    """
    # Create polynomial features
    X_poly = np.ones((X.shape[0], degree + 1))
    for d in range(1, degree + 1):
        X_poly[:, d] = X[:, 0] ** d
    
    # Fit linear regression
    return linear_regression_analysis(X_poly, y)

# Example usage
def create_example_data(n=100, noise_std=0.5):
    """Create synthetic data for demonstration"""
    np.random.seed(42)
    X_raw = np.random.randn(n, 1)
    X = np.column_stack([np.ones(n), X_raw])
    # True relationship: y = 1 + 2x - 0.5x^2 + noise
    y = 1 + 2*X_raw.flatten() - 0.5*X_raw.flatten()**2 + noise_std * np.random.randn(n)
    return X, y

# Example: Quadratic regression
X, y = create_example_data()
X_single = X[:, 1:2]  # Use only first predictor
poly_results = polynomial_regression(X_single, y, degree=2)
print(f"Polynomial R²: {poly_results['R_squared']:.4f}")

# Compare with linear regression
linear_results = linear_regression_analysis(X, y)
print(f"Linear R²: {linear_results['R_squared']:.4f}")
