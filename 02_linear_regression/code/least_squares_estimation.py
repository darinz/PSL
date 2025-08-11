import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def least_squares_estimation(X, y):
    """
    Compute least squares estimates using the normal equation
    
    Parameters:
    X: design matrix (n x (p+1)) including intercept column
    y: response vector (n x 1)
    
    Returns:
    beta_hat: estimated coefficients
    """
    # Check if X^T X is invertible
    XtX = X.T @ X
    if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
        print("Warning: X^T X is not full rank. Solution may not be unique.")
    
    # Compute least squares estimate
    beta_hat = np.linalg.inv(XtX) @ X.T @ y
    return beta_hat

# Example usage
def create_example_data(n=100, p=2, noise_std=0.5):
    """Create synthetic data for demonstration"""
    np.random.seed(42)
    
    # Generate predictors
    X_raw = np.random.randn(n, p)
    
    # Add intercept column
    X = np.column_stack([np.ones(n), X_raw])
    
    # True coefficients
    beta_true = np.array([1.0, 2.0, -1.5])
    
    # Generate response
    y = X @ beta_true + noise_std * np.random.randn(n)
    
    return X, y, beta_true

# Generate data and fit model
X, y, beta_true = create_example_data()
beta_hat = least_squares_estimation(X, y)

print("True coefficients:", beta_true)
print("Estimated coefficients:", beta_hat)
print("Estimation error:", np.linalg.norm(beta_hat - beta_true))
