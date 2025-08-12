"""
R-squared Analysis
==================

This module provides comprehensive analysis of R-squared (coefficient of determination),
including geometric interpretation, variance decomposition, and adjusted R-squared.
"""

import numpy as np
import matplotlib.pyplot as plt

def analyze_r_squared(X, y, y_hat, r):
    """Comprehensive analysis of R-squared"""
    
    print("=== R-squared Analysis ===")
    
    # Compute components
    y_mean = np.mean(y)
    y_centered = y - y_mean
    y_hat_centered = y_hat - y_mean
    
    # Sums of squares
    TSS = np.sum(y_centered**2)  # Total sum of squares
    ESS = np.sum(y_hat_centered**2)  # Explained sum of squares
    RSS = np.sum(r**2)  # Residual sum of squares
    
    print(f"Total Sum of Squares (TSS): {TSS:.4f}")
    print(f"Explained Sum of Squares (ESS): {ESS:.4f}")
    print(f"Residual Sum of Squares (RSS): {RSS:.4f}")
    
    # Verify decomposition
    decomposition_check = abs(TSS - (ESS + RSS))
    print(f"Decomposition check (should be 0): {decomposition_check:.10f}")
    
    # Compute R-squared
    R2_manual = ESS / TSS
    R2_alternative = 1 - RSS / TSS
    
    print(f"\nR-squared (ESS/TSS): {R2_manual:.4f}")
    print(f"R-squared (1 - RSS/TSS): {R2_alternative:.4f}")
    
    # Correlation interpretation
    correlation_y_yhat = np.corrcoef(y, y_hat)[0, 1]
    R2_correlation = correlation_y_yhat**2
    
    print(f"Correlation between y and ŷ: {correlation_y_yhat:.4f}")
    print(f"R-squared from correlation: {R2_correlation:.4f}")
    
    # Geometric interpretation
    norm_y_centered = np.linalg.norm(y_centered)
    norm_yhat_centered = np.linalg.norm(y_hat_centered)
    norm_r = np.linalg.norm(r)
    
    print(f"\nGeometric Interpretation:")
    print(f"||y - ȳ|| = {norm_y_centered:.4f}")
    print(f"||ŷ - ȳ|| = {norm_yhat_centered:.4f}")
    print(f"||r|| = {norm_r:.4f}")
    print(f"R² = (||ŷ - ȳ||/||y - ȳ||)² = {(norm_yhat_centered/norm_y_centered)**2:.4f}")
    
    return {
        'TSS': TSS,
        'ESS': ESS,
        'RSS': RSS,
        'R2': R2_manual,
        'correlation': correlation_y_yhat
    }

def visualize_r_squared(X, y, y_hat, r):
    """Visualize the geometric interpretation of R-squared"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Original vs Fitted
    axes[0,0].scatter(y, y_hat, alpha=0.6)
    axes[0,0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[0,0].set_xlabel('Observed Values (y)')
    axes[0,0].set_ylabel('Fitted Values (ŷ)')
    axes[0,0].set_title('Observed vs Fitted Values')
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals vs Fitted
    axes[0,1].scatter(y_hat, r, alpha=0.6)
    axes[0,1].axhline(y=0, color='red', linestyle='--')
    axes[0,1].set_xlabel('Fitted Values (ŷ)')
    axes[0,1].set_ylabel('Residuals (r)')
    axes[0,1].set_title('Residuals vs Fitted Values')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Variance decomposition
    y_mean = np.mean(y)
    y_centered = y - y_mean
    y_hat_centered = y_hat - y_mean
    
    # Create a bar plot showing the decomposition
    components = ['Total SS', 'Explained SS', 'Residual SS']
    values = [np.sum(y_centered**2), np.sum(y_hat_centered**2), np.sum(r**2)]
    colors = ['blue', 'green', 'red']
    
    bars = axes[1,0].bar(components, values, color=colors, alpha=0.7)
    axes[1,0].set_ylabel('Sum of Squares')
    axes[1,0].set_title('Variance Decomposition')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height,
                      f'{value:.2f}', ha='center', va='bottom')
    
    # Plot 4: R-squared interpretation
    R2 = np.sum(y_hat_centered**2) / np.sum(y_centered**2)
    axes[1,1].pie([R2, 1-R2], labels=[f'Explained\n({R2:.1%})', f'Unexplained\n({1-R2:.1%})'],
                  colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%')
    axes[1,1].set_title('R-squared: Proportion of Variance Explained')
    
    plt.tight_layout()
    plt.show()

def compute_adjusted_r_squared(X, y, y_hat, r):
    """Compute adjusted R-squared"""
    
    n, p_plus_1 = X.shape
    p = p_plus_1 - 1  # Number of predictors (excluding intercept)
    
    # Compute R-squared
    y_mean = np.mean(y)
    TSS = np.sum((y - y_mean)**2)
    RSS = np.sum(r**2)
    R2 = 1 - RSS / TSS
    
    # Compute adjusted R-squared
    R2_adj = 1 - (1 - R2) * (n - 1) / (n - p - 1)
    
    print(f"=== Adjusted R-squared Analysis ===")
    print(f"Sample size (n): {n}")
    print(f"Number of predictors (p): {p}")
    print(f"Degrees of freedom (n-p-1): {n-p-1}")
    print(f"R-squared: {R2:.4f}")
    print(f"Adjusted R-squared: {R2_adj:.4f}")
    
    # Compare with different numbers of predictors
    print(f"\nComparison with different model complexities:")
    for p_test in range(1, min(p+3, n-1)):
        R2_adj_test = 1 - (1 - R2) * (n - 1) / (n - p_test - 1)
        print(f"  p={p_test}: Adjusted R² = {R2_adj_test:.4f}")
    
    return R2_adj

if __name__ == "__main__":
    # Generate sample data for demonstration
    np.random.seed(42)
    n = 50
    p = 2
    
    X_raw = np.random.randn(n, p)
    X = np.column_stack([np.ones(n), X_raw])
    beta_true = np.array([1, 0.5, -0.3])
    y = X @ beta_true + np.random.normal(0, 0.5, n)
    
    # Fit model
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    y_hat = X @ beta_hat
    r = y - y_hat
    
    # Analyze R-squared
    r2_results = analyze_r_squared(X, y, y_hat, r)
    
    # Create R-squared visualization
    visualize_r_squared(X, y, y_hat, r)
    
    # Compute adjusted R-squared
    R2_adj = compute_adjusted_r_squared(X, y, y_hat, r)
