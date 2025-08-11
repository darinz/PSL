import numpy as np
import matplotlib.pyplot as plt

def hat_matrix_demo():
    """Demonstrate the hat matrix and its properties"""
    
    # Create example data
    np.random.seed(42)
    n = 30
    X = np.random.randn(n, 2)
    X = np.column_stack([np.ones(n), X])  # Add intercept
    beta_true = np.array([1.0, 2.0, -1.5])
    y = X @ beta_true + 0.5 * np.random.randn(n)
    
    # Compute hat matrix
    XtX_inv = np.linalg.inv(X.T @ X)
    H = X @ XtX_inv @ X.T
    
    # Verify hat matrix properties
    print("=== Hat Matrix Properties ===")
    print(f"Shape: {H.shape}")
    print(f"Symmetry: {np.allclose(H, H.T)}")
    print(f"Idempotency: {np.allclose(H @ H, H)}")
    print(f"Trace: {np.trace(H):.2f} (should equal p+1 = {X.shape[1]})")
    
    # Compute fitted values using hat matrix
    y_hat = H @ y
    
    # Compute residuals
    residuals = y - y_hat
    
    # Verify orthogonality
    orthogonality = X.T @ residuals
    print(f"Orthogonality check: {np.allclose(orthogonality, 0, atol=1e-10)}")
    
    # Leverage values (diagonal of hat matrix)
    leverage = np.diag(H)
    print(f"Leverage range: [{leverage.min():.4f}, {leverage.max():.4f}]")
    print(f"Average leverage: {leverage.mean():.4f}")
    
    return H, y_hat, residuals, leverage

# Run demonstration
H, y_hat, residuals, leverage = hat_matrix_demo()
