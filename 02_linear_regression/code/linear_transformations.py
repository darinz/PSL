"""
Linear Transformations
=====================

This module demonstrates the effect of linear transformations on linear regression,
showing how different transformations affect coefficients while preserving fitted values.
"""

import numpy as np
import matplotlib.pyplot as plt

def demonstrate_linear_transformations():
    """Demonstrate the effect of linear transformations on regression"""
    
    # Generate data
    np.random.seed(42)
    n = 100
    p = 2
    
    X_raw = np.random.randn(n, p)
    X = np.column_stack([np.ones(n), X_raw])
    beta_true = np.array([1, 2, -1.5])
    y = X @ beta_true + np.random.normal(0, 0.1, n)
    
    print("=== Linear Transformations Analysis ===")
    print(f"Original coefficients: {beta_true}")
    
    # 1. Scaling transformation
    scale_factor = 2.0
    A_scale = np.array([[1, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
    X_scaled = X @ A_scale
    
    # Fit both models
    beta_orig = np.linalg.inv(X.T @ X) @ X.T @ y
    beta_scaled = np.linalg.inv(X_scaled.T @ X_scaled) @ X_scaled.T @ y
    
    print(f"\n1. Scaling Transformation (factor = {scale_factor})")
    print(f"Original coefficients: {beta_orig}")
    print(f"Scaled coefficients: {beta_scaled}")
    print(f"Ratio (should be {scale_factor}): {beta_orig[1] / beta_scaled[1]:.6f}")
    
    # Check invariance of fitted values
    y_hat_orig = X @ beta_orig
    y_hat_scaled = X_scaled @ beta_scaled
    invariance_check = np.linalg.norm(y_hat_orig - y_hat_scaled)
    print(f"Fitted values invariance check: {invariance_check:.10f}")
    
    # 2. Centering transformation
    x1_mean = np.mean(X[:, 1])
    x2_mean = np.mean(X[:, 2])
    A_center = np.array([[1, -x1_mean, -x2_mean], [0, 1, 0], [0, 0, 1]])
    X_centered = X @ A_center
    
    beta_centered = np.linalg.inv(X_centered.T @ X_centered) @ X_centered.T @ y
    
    print(f"\n2. Centering Transformation")
    print(f"Original coefficients: {beta_orig}")
    print(f"Centered coefficients: {beta_centered}")
    print(f"Intercept change: {beta_centered[0] - beta_orig[0]:.6f}")
    print(f"Expected intercept change: {beta_orig[1] * x1_mean + beta_orig[2] * x2_mean:.6f}")
    
    # 3. Standardization transformation
    x1_std = np.std(X[:, 1])
    x2_std = np.std(X[:, 2])
    A_standardize = np.array([[1, -x1_mean/x1_std, -x2_mean/x2_std], 
                              [0, 1/x1_std, 0], 
                              [0, 0, 1/x2_std]])
    X_standardized = X @ A_standardize
    
    beta_standardized = np.linalg.inv(X_standardized.T @ X_standardized) @ X_standardized.T @ y
    
    print(f"\n3. Standardization Transformation")
    print(f"Original coefficients: {beta_orig}")
    print(f"Standardized coefficients: {beta_standardized}")
    print(f"Slope scaling check: {beta_orig[1] * x1_std:.6f} vs {beta_standardized[1]:.6f}")
    
    return {
        'original': beta_orig,
        'scaled': beta_scaled,
        'centered': beta_centered,
        'standardized': beta_standardized,
        'X': X,
        'y': y
    }

def visualize_transformations(X, y, transformation_results):
    """Visualize the effect of different transformations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original data
    axes[0,0].scatter(X[:, 1], y, alpha=0.6)
    axes[0,0].set_xlabel('X1')
    axes[0,0].set_ylabel('Y')
    axes[0,0].set_title('Original Data')
    axes[0,0].grid(True, alpha=0.3)
    
    # Scaled data
    scale_factor = 2.0
    X_scaled = X.copy()
    X_scaled[:, 1] *= scale_factor
    axes[0,1].scatter(X_scaled[:, 1], y, alpha=0.6)
    axes[0,1].set_xlabel('X1 (scaled)')
    axes[0,1].set_ylabel('Y')
    axes[0,1].set_title('Scaled Data')
    axes[0,1].grid(True, alpha=0.3)
    
    # Centered data
    X_centered = X.copy()
    X_centered[:, 1] -= np.mean(X[:, 1])
    axes[1,0].scatter(X_centered[:, 1], y, alpha=0.6)
    axes[1,0].set_xlabel('X1 (centered)')
    axes[1,0].set_ylabel('Y')
    axes[1,0].set_title('Centered Data')
    axes[1,0].grid(True, alpha=0.3)
    
    # Standardized data
    X_standardized = X.copy()
    X_standardized[:, 1] = (X[:, 1] - np.mean(X[:, 1])) / np.std(X[:, 1])
    axes[1,1].scatter(X_standardized[:, 1], y, alpha=0.6)
    axes[1,1].set_xlabel('X1 (standardized)')
    axes[1,1].set_ylabel('Y')
    axes[1,1].set_title('Standardized Data')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Demonstrate linear transformations
    transformation_results = demonstrate_linear_transformations()
    
    # Create transformation visualization
    visualize_transformations(transformation_results['X'], transformation_results['y'], transformation_results)
