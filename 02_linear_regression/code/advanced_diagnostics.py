"""
Advanced Diagnostics
===================

This module provides advanced diagnostic measures for linear regression,
including leverage, studentized residuals, and Cook's distance.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def advanced_diagnostics(X, y, y_hat, r):
    """Compute advanced diagnostic measures"""
    
    print("=== Advanced Diagnostic Measures ===")
    
    n, p_plus_1 = X.shape
    p = p_plus_1 - 1
    
    # Hat matrix
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    leverage = np.diag(H)
    
    print(f"Leverage statistics:")
    print(f"  Mean leverage: {np.mean(leverage):.4f}")
    print(f"  Expected leverage: {(p+1)/n:.4f}")
    print(f"  Max leverage: {np.max(leverage):.4f}")
    print(f"  Min leverage: {np.min(leverage):.4f}")
    
    # High leverage threshold
    threshold = 2 * (p+1) / n
    high_leverage = leverage > threshold
    print(f"  High leverage observations: {np.sum(high_leverage)}")
    
    # Standardized residuals
    sigma_hat = np.sqrt(np.sum(r**2) / (n - p - 1))
    standardized_residuals = r / (sigma_hat * np.sqrt(1 - leverage))
    
    print(f"\nStandardized residuals:")
    print(f"  Mean: {np.mean(standardized_residuals):.4f}")
    print(f"  Std: {np.std(standardized_residuals):.4f}")
    print(f"  Max: {np.max(standardized_residuals):.4f}")
    print(f"  Min: {np.min(standardized_residuals):.4f}")
    
    # Studentized residuals (externally studentized)
    def compute_studentized_residuals():
        studentized = np.zeros(n)
        for i in range(n):
            # Remove observation i
            X_i = np.delete(X, i, axis=0)
            y_i = np.delete(y, i)
            
            # Fit model without observation i
            beta_i = np.linalg.inv(X_i.T @ X_i) @ X_i.T @ y_i
            y_hat_i = X_i @ beta_i
            r_i = y_i - y_hat_i
            
            # Compute sigma without observation i
            sigma_i = np.sqrt(np.sum(r_i**2) / (n - p - 2))
            
            # Compute studentized residual
            x_i = X[i, :].reshape(1, -1)
            y_hat_i_full = x_i @ beta_i
            studentized[i] = (y[i] - y_hat_i_full) / (sigma_i * np.sqrt(1 + x_i @ np.linalg.inv(X_i.T @ X_i) @ x_i.T))
        
        return studentized
    
    studentized_residuals = compute_studentized_residuals()
    
    print(f"\nStudentized residuals:")
    print(f"  Mean: {np.mean(studentized_residuals):.4f}")
    print(f"  Std: {np.std(studentized_residuals):.4f}")
    print(f"  Max: {np.max(studentized_residuals):.4f}")
    print(f"  Min: {np.min(studentized_residuals):.4f}")
    
    # Cook's distance
    def compute_cooks_distance():
        cooks = np.zeros(n)
        sigma_hat = np.sqrt(np.sum(r**2) / (n - p - 1))
        
        for i in range(n):
            # Remove observation i
            X_i = np.delete(X, i, axis=0)
            y_i = np.delete(y, i)
            
            # Fit model without observation i
            beta_i = np.linalg.inv(X_i.T @ X_i) @ X_i.T @ y_i
            
            # Compute Cook's distance
            beta_diff = beta_hat - beta_i
            cooks[i] = (beta_diff.T @ X.T @ X @ beta_diff) / ((p+1) * sigma_hat**2)
        
        return cooks
    
    cooks_distance = compute_cooks_distance()
    
    print(f"\nCook's distance:")
    print(f"  Mean: {np.mean(cooks_distance):.4f}")
    print(f"  Max: {np.max(cooks_distance):.4f}")
    print(f"  Threshold (4/n): {4/n:.4f}")
    
    high_influence = cooks_distance > 4/n
    print(f"  High influence observations: {np.sum(high_influence)}")
    
    return {
        'leverage': leverage,
        'standardized_residuals': standardized_residuals,
        'studentized_residuals': studentized_residuals,
        'cooks_distance': cooks_distance,
        'high_leverage': high_leverage,
        'high_influence': high_influence
    }

def plot_advanced_diagnostics(X, y, diagnostics):
    """Plot advanced diagnostic measures"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Leverage vs residuals
    axes[0,0].scatter(diagnostics['leverage'], diagnostics['standardized_residuals'], alpha=0.6)
    axes[0,0].axhline(y=0, color='red', linestyle='--')
    axes[0,0].axvline(x=2*(X.shape[1])/X.shape[0], color='red', linestyle='--')
    axes[0,0].set_xlabel('Leverage')
    axes[0,0].set_ylabel('Standardized Residuals')
    axes[0,0].set_title('Leverage vs Standardized Residuals')
    axes[0,0].grid(True, alpha=0.3)
    
    # Cook's distance
    axes[0,1].plot(range(len(diagnostics['cooks_distance'])), diagnostics['cooks_distance'], 'o-')
    axes[0,1].axhline(y=4/len(y), color='red', linestyle='--', label='4/n threshold')
    axes[0,1].set_xlabel('Observation Index')
    axes[0,1].set_ylabel("Cook's Distance")
    axes[0,1].set_title("Cook's Distance")
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Studentized residuals
    axes[1,0].scatter(range(len(diagnostics['studentized_residuals'])), 
                     diagnostics['studentized_residuals'], alpha=0.6)
    axes[1,0].axhline(y=0, color='red', linestyle='--')
    axes[1,0].axhline(y=2, color='orange', linestyle='--', label='±2')
    axes[1,0].axhline(y=-2, color='orange', linestyle='--')
    axes[1,0].set_xlabel('Observation Index')
    axes[1,0].set_ylabel('Studentized Residuals')
    axes[1,0].set_title('Studentized Residuals')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Q-Q plot of studentized residuals
    stats.probplot(diagnostics['studentized_residuals'], dist="norm", plot=axes[1,1])
    axes[1,1].set_title('Q-Q Plot of Studentized Residuals')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Generate sample data for demonstration
    np.random.seed(42)
    n = 50
    p = 2
    
    X_raw = np.random.randn(n, p)
    X = np.column_stack([np.ones(n), X_raw])
    beta_true = np.array([1, 0.5, -0.3])
    y = X @ beta_true + np.random.normal(0, 0.5, n)
    
    # Add some outliers for demonstration
    y[0] += 3  # High leverage point
    y[10] += 2  # Influential point
    
    # Fit model
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    y_hat = X @ beta_hat
    r = y - y_hat
    
    # Compute advanced diagnostics
    diagnostics = advanced_diagnostics(X, y, y_hat, r)
    
    # Create diagnostic plots
    plot_advanced_diagnostics(X, y, diagnostics)
