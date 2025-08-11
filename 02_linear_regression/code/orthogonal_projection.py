import numpy as np
import matplotlib.pyplot as plt

def orthogonal_projection_demo():
    """Demonstrate orthogonal projection in linear regression"""
    
    # Create example data
    np.random.seed(42)
    n = 50
    X = np.random.randn(n, 2)
    X = np.column_stack([np.ones(n), X])  # Add intercept
    beta_true = np.array([1.0, 2.0, -1.5])
    y = X @ beta_true + 0.5 * np.random.randn(n)
    
    # Compute least squares estimate
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    
    # Compute fitted values (projection)
    y_hat = X @ beta_hat
    
    # Compute residuals (orthogonal to column space)
    residuals = y - y_hat
    
    # Verify orthogonality
    orthogonality_check = X.T @ residuals
    print("Orthogonality check (should be close to zero):")
    print(orthogonality_check)
    
    # Verify projection properties
    print(f"\nProjection properties:")
    print(f"y_hat is in column space of X: {np.linalg.matrix_rank(np.column_stack([X, y_hat])) == np.linalg.matrix_rank(X)}")
    print(f"residuals orthogonal to X: {np.allclose(orthogonality_check, 0, atol=1e-10)}")
    
    return X, y, y_hat, residuals

# Run demonstration
X, y, y_hat, residuals = orthogonal_projection_demo()
