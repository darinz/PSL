import numpy as np
import matplotlib.pyplot as plt

def least_squares_estimation(X, y):
    """Compute least squares estimates using the normal equation"""
    XtX = X.T @ X
    if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
        print("Warning: X^T X is not full rank. Solution may not be unique.")
    beta_hat = np.linalg.inv(XtX) @ X.T @ y
    return beta_hat

def compute_vif(X):
    """
    Compute Variance Inflation Factors
    
    Parameters:
    X: design matrix (with intercept)
    
    Returns:
    vif_values: VIF for each predictor
    """
    n, p = X.shape
    vif_values = []
    
    for j in range(1, p):  # Skip intercept
        # Regress predictor j on all other predictors
        X_j = X[:, j].reshape(-1, 1)
        X_others = np.delete(X, j, axis=1)
        
        # Fit regression
        beta_j = least_squares_estimation(X_others, X_j.flatten())
        y_j_hat = X_others @ beta_j
        rss_j = np.sum((X_j.flatten() - y_j_hat)**2)
        tss_j = np.sum((X_j.flatten() - np.mean(X_j))**2)
        
        # Compute VIF
        vif = 1 / (1 - (1 - rss_j/tss_j))
        vif_values.append(vif)
    
    return vif_values

# Example usage
def create_example_data(n=100, p=3, noise_std=0.5):
    """Create synthetic data for demonstration"""
    np.random.seed(42)
    X_raw = np.random.randn(n, p)
    X = np.column_stack([np.ones(n), X_raw])
    beta_true = np.array([1.0, 2.0, -1.5, 0.5])
    y = X @ beta_true + noise_std * np.random.randn(n)
    return X, y, beta_true

# Check for multicollinearity
X, y, beta_true = create_example_data()
vif_values = compute_vif(X)

print("=== Multicollinearity Check ===")
for i, vif in enumerate(vif_values):
    print(f"VIF for predictor {i+1}: {vif:.2f}")
    if vif > 10:
        print(f"  Warning: High multicollinearity detected!")
