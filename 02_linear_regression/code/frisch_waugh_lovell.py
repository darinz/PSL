"""
Frisch-Waugh-Lovell Theorem Implementation
==========================================

This module demonstrates the Frisch-Waugh-Lovell theorem, which provides
an elegant way to understand how coefficients are computed in multiple regression
by decomposing the multiple regression coefficient into a series of simple regressions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def frisch_waugh_lovell(X, y, k):
    """
    Implement Frisch-Waugh-Lovell theorem for coefficient k
    
    Parameters:
    X: design matrix (n x p)
    y: response variable (n x 1)
    k: index of coefficient to compute (0-based)
    
    Returns:
    beta_k: coefficient estimate
    y_resid: residuals from step 1
    xk_resid: residuals from step 2
    """
    n, p = X.shape
    
    # Step 1: Regress y on all predictors except X_k
    X_minus_k = np.delete(X, k, axis=1)
    model_y = LinearRegression()
    model_y.fit(X_minus_k, y)
    y_resid = y - model_y.predict(X_minus_k)
    
    # Step 2: Regress X_k on all other predictors
    X_k = X[:, k]
    model_xk = LinearRegression()
    model_xk.fit(X_minus_k, X_k)
    xk_resid = X_k - model_xk.predict(X_minus_k)
    
    # Step 3: Regress y_resid on xk_resid
    model_final = LinearRegression()
    model_final.fit(xk_resid.reshape(-1, 1), y_resid)
    beta_k = model_final.coef_[0]
    
    return beta_k, y_resid, xk_resid

def demonstrate_fwl_theorem():
    """Demonstrate the Frisch-Waugh-Lovell theorem"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate correlated data
    n = 200
    X1 = np.random.normal(0, 1, n)
    X2 = 0.6 * X1 + np.random.normal(0, 0.8, n)  # Correlated with X1
    X3 = np.random.normal(0, 1, n)  # Independent
    
    # True model
    beta0_true = 2.0
    beta1_true = 1.5
    beta2_true = -0.8
    beta3_true = 0.4
    
    y = beta0_true + beta1_true * X1 + beta2_true * X2 + beta3_true * X3 + np.random.normal(0, 0.5, n)
    
    # Create design matrix
    X = np.column_stack([X1, X2, X3])
    feature_names = ['X1', 'X2', 'X3']
    
    print("=== TRUE MODEL ===")
    print(f"Y = {beta0_true} + {beta1_true}*X1 + {beta2_true}*X2 + {beta3_true}*X3 + ε")
    
    # Standard multiple regression
    print("\n=== STANDARD MULTIPLE REGRESSION ===")
    model_standard = LinearRegression()
    model_standard.fit(X, y)
    standard_coefs = model_standard.coef_
    standard_intercept = model_standard.intercept_
    
    print("Standard multiple regression results:")
    for i, (name, coef) in enumerate(zip(feature_names, standard_coefs)):
        print(f"  {name}: {coef:.4f}")
    print(f"  Intercept: {standard_intercept:.4f}")
    
    # Frisch-Waugh-Lovell for each coefficient
    print("\n=== FRISCH-WAUGH-LOVELL DECOMPOSITION ===")
    
    # Apply FWL for each coefficient
    fwl_coefs = []
    for k in range(X.shape[1]):
        beta_k, y_resid, xk_resid = frisch_waugh_lovell(X, y, k)
        fwl_coefs.append(beta_k)
        
        print(f"\nFWL for {feature_names[k]}:")
        print(f"  Step 1: Regress Y on {[name for i, name in enumerate(feature_names) if i != k]}")
        print(f"  Step 2: Regress {feature_names[k]} on {[name for i, name in enumerate(feature_names) if i != k]}")
        print(f"  Step 3: Regress Y_resid on {feature_names[k]}_resid")
        print(f"  Coefficient: {beta_k:.4f}")
        print(f"  Standard coefficient: {standard_coefs[k]:.4f}")
        print(f"  Difference: {abs(beta_k - standard_coefs[k]):.8f}")
    
    # Verification: all coefficients should be identical
    print("\n=== VERIFICATION ===")
    print("Coefficient comparison:")
    comparison_df = pd.DataFrame({
        'Feature': feature_names,
        'Standard': standard_coefs,
        'FWL': fwl_coefs,
        'Difference': np.abs(np.array(standard_coefs) - np.array(fwl_coefs))
    })
    print(comparison_df.to_string(index=False))
    
    return X, y, feature_names, standard_coefs, fwl_coefs

def visualize_fwl_process(X, y, feature_names):
    """Visualize the Frisch-Waugh-Lovell process"""
    
    # Visualization: Show the partialling out process for X1
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original relationships
    axes[0, 0].scatter(X[:, 0], y, alpha=0.6)
    axes[0, 0].set_xlabel('X1')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].set_title('Original: Y vs X1')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(X[:, 1], y, alpha=0.6)
    axes[0, 1].set_xlabel('X2')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].set_title('Original: Y vs X2')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].scatter(X[:, 0], X[:, 1], alpha=0.6)
    axes[0, 2].set_xlabel('X1')
    axes[0, 2].set_ylabel('X2')
    axes[0, 2].set_title('Correlation: X1 vs X2')
    axes[0, 2].grid(True, alpha=0.3)
    
    # FWL residuals for X1
    _, y_resid_x1, x1_resid = frisch_waugh_lovell(X, y, 0)
    
    axes[1, 0].scatter(x1_resid, y_resid_x1, alpha=0.6)
    axes[1, 0].set_xlabel('X1_resid (X1 partialled out)')
    axes[1, 0].set_ylabel('Y_resid (Y partialled out)')
    axes[1, 0].set_title('FWL: Y_resid vs X1_resid')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Fit line to show the relationship
    model_fwl = LinearRegression()
    model_fwl.fit(x1_resid.reshape(-1, 1), y_resid_x1)
    x1_range = np.linspace(x1_resid.min(), x1_resid.max(), 100)
    y_pred_fwl = model_fwl.predict(x1_range.reshape(-1, 1))
    axes[1, 0].plot(x1_range, y_pred_fwl, 'r-', linewidth=2)
    
    # Check orthogonality
    axes[1, 1].scatter(x1_resid, X[:, 1], alpha=0.6)
    axes[1, 1].set_xlabel('X1_resid')
    axes[1, 1].set_ylabel('X2')
    axes[1, 1].set_title('Orthogonality: X1_resid vs X2')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Check orthogonality with X3
    axes[1, 2].scatter(x1_resid, X[:, 2], alpha=0.6)
    axes[1, 2].set_xlabel('X1_resid')
    axes[1, 2].set_ylabel('X3')
    axes[1, 2].set_title('Orthogonality: X1_resid vs X3')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return y_resid_x1, x1_resid

def verify_orthogonality(X, y_resid_x1, x1_resid, feature_names):
    """Verify orthogonality of FWL residuals"""
    
    # Verify orthogonality numerically
    print("\n=== ORTHOGONALITY VERIFICATION ===")
    print("Correlation between X1_resid and other predictors:")
    for i, name in enumerate(feature_names[1:], 1):
        corr = np.corrcoef(x1_resid, X[:, i])[0, 1]
        print(f"  X1_resid vs {name}: {corr:.6f}")
    
    # Theoretical verification
    fwl_coefs = []
    for k in range(X.shape[1]):
        beta_k, _, _ = frisch_waugh_lovell(X, y, k)
        fwl_coefs.append(beta_k)
    
    print(f"\nTheoretical FWL coefficient for X1: {fwl_coefs[0]:.4f}")
    print(f"Direct calculation: Cov(Y_resid, X1_resid) / Var(X1_resid)")
    cov_ratio = np.cov(y_resid_x1, x1_resid)[0, 1] / np.var(x1_resid)
    print(f"  = {cov_ratio:.4f}")

if __name__ == "__main__":
    # Demonstrate FWL theorem
    X, y, feature_names, standard_coefs, fwl_coefs = demonstrate_fwl_theorem()
    
    # Visualize the process
    y_resid_x1, x1_resid = visualize_fwl_process(X, y, feature_names)
    
    # Verify orthogonality
    verify_orthogonality(X, y_resid_x1, x1_resid, feature_names)
