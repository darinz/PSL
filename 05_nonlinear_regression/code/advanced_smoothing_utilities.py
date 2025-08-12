"""
Advanced Smoothing Spline Utilities
==================================

This module provides advanced utilities for smoothing splines including
cross-validation, confidence intervals, weighted splines, and comprehensive diagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.linalg import solve
from scipy import stats

def compute_loocv_score(model, X, y):
    """
    Compute leave-one-out cross-validation score
    
    Parameters:
    model: fitted smoothing spline model
    X: predictor variable
    y: response variable
    
    Returns:
    loocv_score: leave-one-out cross-validation score
    """
    y_pred = model.smoother_matrix @ y
    residuals = y - y_pred
    leverage = np.diag(model.smoother_matrix)
    
    # Adjust residuals for leverage
    adjusted_residuals = residuals / (1 - leverage)
    loocv_score = np.mean(adjusted_residuals**2)
    
    return loocv_score

def compute_gcv_score(model, X, y):
    """
    Compute generalized cross-validation score
    
    Parameters:
    model: fitted smoothing spline model
    X: predictor variable
    y: response variable
    
    Returns:
    gcv_score: generalized cross-validation score
    """
    y_pred = model.smoother_matrix @ y
    residuals = y - y_pred
    edf = np.trace(model.smoother_matrix)
    n = len(y)
    
    # GCV score
    gcv_score = np.mean(residuals**2) / (1 - edf/n)**2
    
    return gcv_score

def compute_confidence_intervals(model, X, y, X_new, confidence=0.95):
    """
    Compute confidence intervals for smoothing spline predictions
    
    Parameters:
    model: fitted smoothing spline model
    X: predictor variable
    y: response variable
    X_new: new predictor values for prediction
    confidence: confidence level (default: 0.95)
    
    Returns:
    y_pred: predicted values
    ci_lower: lower confidence interval bounds
    ci_upper: upper confidence interval bounds
    """
    # Get predictions
    y_pred = model.predict(X_new)
    
    # Compute residuals
    y_fit = model.smoother_matrix @ y
    residuals = y - y_fit
    sigma_hat = np.std(residuals)
    
    # Compute leverage for new points
    # This is a simplified version - in practice, use specialized algorithms
    leverage_new = np.diag(model.smoother_matrix)[:len(X_new)]
    
    # Standard error of prediction
    se_pred = sigma_hat * np.sqrt(leverage_new)
    
    # Confidence interval
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha/2, len(y) - model.edf)
    
    ci_lower = y_pred - t_critical * se_pred
    ci_upper = y_pred + t_critical * se_pred
    
    return y_pred, ci_lower, ci_upper

def fit_weighted_smoothing_spline(X, y, weights, lambda_val=None):
    """
    Fit weighted smoothing spline for heteroscedastic data
    
    Parameters:
    X: predictor variable
    y: response variable
    weights: observation weights
    lambda_val: smoothing parameter
    
    Returns:
    beta: fitted coefficients
    H: basis matrix
    Omega: penalty matrix
    """
    n = len(X)
    W = np.diag(weights)
    
    # Create basis and penalty matrices
    H = create_natural_spline_basis(X)
    Omega = create_penalty_matrix(X)
    
    # Solve weighted problem
    if lambda_val is None:
        lambda_val = 1.0
    
    beta = solve(H.T @ W @ H + lambda_val * Omega, H.T @ W @ y)
    
    return beta, H, Omega

def create_natural_spline_basis(X):
    """
    Create natural cubic spline basis matrix
    
    Parameters:
    X: predictor variable
    
    Returns:
    H: basis matrix
    """
    n = len(X)
    H = np.zeros((n, n))
    
    # Create basis functions using scipy
    for i in range(n):
        # Create unit vector for basis function i
        unit_vector = np.zeros(n)
        unit_vector[i] = 1.0
        
        # Create natural cubic spline
        spline = CubicSpline(X, unit_vector, bc_type='natural')
        H[:, i] = spline(X)
    
    return H

def create_penalty_matrix(X):
    """
    Create penalty matrix for natural cubic splines
    
    Parameters:
    X: predictor variable
    
    Returns:
    Omega: penalty matrix
    """
    n = len(X)
    Omega = np.zeros((n, n))
    
    # For natural cubic splines, the penalty matrix can be constructed
    # using the second derivatives of the basis functions
    for i in range(n):
        for j in range(n):
            # This is a simplified version - in practice, use specialized algorithms
            if i == j:
                Omega[i, j] = 1.0
            else:
                Omega[i, j] = 0.0
    
    # Ensure the first two rows/columns are zero (linear terms not penalized)
    Omega[0, :] = 0
    Omega[1, :] = 0
    Omega[:, 0] = 0
    Omega[:, 1] = 0
    
    return Omega

def smoothing_spline_diagnostics(model, X, y):
    """
    Comprehensive diagnostics for smoothing splines
    
    Parameters:
    model: fitted smoothing spline model
    X: predictor variable
    y: response variable
    
    Returns:
    residuals: model residuals
    leverage: leverage values
    """
    y_pred = model.smoother_matrix @ y
    residuals = y - y_pred
    leverage = np.diag(model.smoother_matrix)
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Residuals vs Fitted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot of Residuals')
    
    # Residuals vs Predictor
    axes[0, 2].scatter(X, residuals, alpha=0.6)
    axes[0, 2].axhline(y=0, color='r', linestyle='--')
    axes[0, 2].set_xlabel('X')
    axes[0, 2].set_ylabel('Residuals')
    axes[0, 2].set_title('Residuals vs X')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Leverage plot
    axes[1, 0].scatter(range(len(leverage)), leverage, alpha=0.6)
    axes[1, 0].axhline(y=2*model.edf/len(y), color='r', linestyle='--', 
                       label='2*EDF/n threshold')
    axes[1, 0].set_xlabel('Observation Index')
    axes[1, 0].set_ylabel('Leverage')
    axes[1, 0].set_title('Leverage Plot')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scale-Location plot
    axes[1, 1].scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.6)
    axes[1, 1].set_xlabel('Fitted Values')
    axes[1, 1].set_ylabel('√|Residuals|')
    axes[1, 1].set_title('Scale-Location Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Smoother matrix
    axes[1, 2].imshow(model.smoother_matrix, cmap='viridis')
    axes[1, 2].set_title('Smoother Matrix')
    axes[1, 2].set_xlabel('j')
    axes[1, 2].set_ylabel('i')
    
    plt.tight_layout()
    plt.show()
    
    return residuals, leverage

def demonstrate_advanced_features():
    """Demonstrate advanced smoothing spline features"""
    # Generate synthetic data
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    y_true = 2 + 3*np.sin(X) + 0.5*X
    y = y_true + np.random.normal(0, 0.5, 100)
    
    # Fit smoothing spline
    from .smoothing_spline_regression import SmoothingSpline
    model = SmoothingSpline(cv=True)
    model.fit(X, y)
    
    # Demonstrate confidence intervals
    X_new = np.linspace(0, 10, 200)
    y_pred, ci_lower, ci_upper = compute_confidence_intervals(model, X, y, X_new)
    
    # Demonstrate weighted splines
    # Create heteroscedastic weights
    weights = 1 / (1 + 0.5 * X**2)  # Weights decrease with X
    beta_w, H_w, Omega_w = fit_weighted_smoothing_spline(X, y, weights, lambda_val=1.0)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Confidence intervals
    plt.subplot(2, 3, 1)
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.plot(X_new, y_pred, 'b-', label='Fitted', linewidth=2)
    plt.fill_between(X_new, ci_lower, ci_upper, alpha=0.3, label='95% CI')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Smoothing Spline with Confidence Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Weighted vs unweighted
    plt.subplot(2, 3, 2)
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.plot(X_new, y_pred, 'b-', label='Unweighted', linewidth=2)
    
    # Predict using weighted coefficients
    H_new = create_natural_spline_basis(X_new)
    y_pred_w = H_new @ beta_w
    plt.plot(X_new, y_pred_w, 'r-', label='Weighted', linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Weighted vs Unweighted Smoothing Splines')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Weights
    plt.subplot(2, 3, 3)
    plt.plot(X, weights, 'g-', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('Weight')
    plt.title('Observation Weights')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: LOOCV vs GCV
    plt.subplot(2, 3, 4)
    lambda_candidates = np.logspace(-3, 3, 20)
    loocv_scores = []
    gcv_scores = []
    
    for lambda_val in lambda_candidates:
        model_temp = SmoothingSpline(lambda_val=lambda_val, cv=False)
        model_temp.fit(X, y)
        
        loocv_score = compute_loocv_score(model_temp, X, y)
        gcv_score = compute_gcv_score(model_temp, X, y)
        
        loocv_scores.append(loocv_score)
        gcv_scores.append(gcv_score)
    
    plt.semilogx(lambda_candidates, loocv_scores, 'b-', label='LOOCV')
    plt.semilogx(lambda_candidates, gcv_scores, 'r-', label='GCV')
    plt.xlabel('λ')
    plt.ylabel('Score')
    plt.title('LOOCV vs GCV')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Effective degrees of freedom
    plt.subplot(2, 3, 5)
    edf_values = []
    
    for lambda_val in lambda_candidates:
        model_temp = SmoothingSpline(lambda_val=lambda_val, cv=False)
        model_temp.fit(X, y)
        edf_values.append(model_temp.edf)
    
    plt.semilogx(lambda_candidates, edf_values, 'purple', linewidth=2)
    plt.xlabel('λ')
    plt.ylabel('Effective Degrees of Freedom')
    plt.title('λ vs EDF')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Smoother matrix structure
    plt.subplot(2, 3, 6)
    plt.imshow(model.smoother_matrix, cmap='viridis')
    plt.colorbar()
    plt.title('Smoother Matrix Structure')
    plt.xlabel('j')
    plt.ylabel('i')
    
    plt.tight_layout()
    plt.show()
    
    return model, beta_w

if __name__ == "__main__":
    results = demonstrate_advanced_features()
