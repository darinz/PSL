"""
Projection Analysis
==================

This module demonstrates projection and orthogonality in linear regression,
showing how least squares finds the orthogonal projection of the response
vector onto the column space of the design matrix.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def demonstrate_projection():
    """Demonstrate projection and orthogonality in linear regression"""
    
    # Generate synthetic data
    np.random.seed(42)
    n = 10
    p = 2
    
    # Create design matrix
    X_raw = np.random.randn(n, p)
    X = np.column_stack([np.ones(n), X_raw])  # Add intercept
    
    # True coefficients
    beta_true = np.array([2, 1, -0.5])
    
    # Generate response with noise
    y = X @ beta_true + np.random.normal(0, 0.1, n)
    
    # Least squares solution
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    y_hat = X @ beta_hat
    r = y - y_hat
    
    print("=== Projection Analysis ===")
    print(f"True coefficients: {beta_true}")
    print(f"Estimated coefficients: {beta_hat}")
    print(f"Estimation error: {np.linalg.norm(beta_hat - beta_true):.6f}")
    
    # Check orthogonality
    orthogonality = y_hat @ r
    print(f"\nOrthogonality check (should be close to 0): {orthogonality:.10f}")
    
    # Check Pythagorean theorem
    norm_y_sq = np.sum(y**2)
    norm_yhat_sq = np.sum(y_hat**2)
    norm_r_sq = np.sum(r**2)
    pythagorean_check = abs(norm_y_sq - (norm_yhat_sq + norm_r_sq))
    print(f"Pythagorean theorem check: {pythagorean_check:.10f}")
    
    # Visualize in 3D (if n=3) or show relationships
    if n == 3:
        fig = plt.figure(figsize=(15, 5))
        
        # Plot 1: Original data and fitted values
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(X[:, 1], X[:, 2], y, color='blue', s=50, label='Observed')
        ax1.scatter(X[:, 1], X[:, 2], y_hat, color='red', s=50, label='Fitted')
        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')
        ax1.set_zlabel('Y')
        ax1.set_title('Observed vs Fitted Values')
        ax1.legend()
        
        # Plot 2: Residuals
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(X[:, 1], X[:, 2], r, color='green', s=50)
        ax2.set_xlabel('X1')
        ax2.set_ylabel('X2')
        ax2.set_zlabel('Residuals')
        ax2.set_title('Residuals')
        
        # Plot 3: Orthogonality
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(y_hat, r, np.zeros_like(r), color='purple', s=50)
        ax3.set_xlabel('Fitted Values')
        ax3.set_ylabel('Residuals')
        ax3.set_zlabel('Z')
        ax3.set_title('Orthogonality: Fitted vs Residuals')
        
        plt.tight_layout()
        plt.show()
    
    return X, y, y_hat, r, beta_hat

def analyze_projection_properties(X, y, y_hat, r):
    """Analyze key properties of the projection"""
    
    print("\n=== Projection Properties Analysis ===")
    
    # 1. Residuals sum to zero (if intercept included)
    residual_sum = np.sum(r)
    print(f"Sum of residuals: {residual_sum:.10f}")
    
    # 2. Residuals are orthogonal to all predictors
    for j in range(X.shape[1]):
        orthogonality_j = r @ X[:, j]
        print(f"Residuals orthogonal to predictor {j}: {orthogonality_j:.10f}")
    
    # 3. Variance decomposition
    y_mean = np.mean(y)
    TSS = np.sum((y - y_mean)**2)  # Total sum of squares
    RSS = np.sum(r**2)  # Residual sum of squares
    ESS = TSS - RSS  # Explained sum of squares
    
    print(f"\nVariance Decomposition:")
    print(f"Total SS: {TSS:.4f}")
    print(f"Explained SS: {ESS:.4f}")
    print(f"Residual SS: {RSS:.4f}")
    print(f"R² = {ESS/TSS:.4f}")
    
    # 4. Projection matrix properties
    H = X @ np.linalg.inv(X.T @ X) @ X.T  # Hat matrix
    print(f"\nProjection Matrix (Hat Matrix) Properties:")
    print(f"Trace(H) = {np.trace(H):.4f} (should equal p+1)")
    print(f"H is symmetric: {np.allclose(H, H.T)}")
    print(f"H is idempotent: {np.allclose(H @ H, H)}")

if __name__ == "__main__":
    # Demonstrate projection
    X, y, y_hat, r, beta_hat = demonstrate_projection()
    
    # Analyze projection properties
    analyze_projection_properties(X, y, y_hat, r)
