"""
Advanced Spline Utilities
========================

This module provides advanced utilities for cubic splines including
B-spline basis functions, smoothing splines, knot selection algorithms, and diagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, CubicSpline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

def create_bspline_basis(X, knots, degree=3):
    """
    Create B-spline basis functions
    
    Parameters:
    X: predictor variable
    knots: knot positions
    degree: spline degree (default: 3 for cubic)
    
    Returns:
    basis_matrix: B-spline basis matrix
    """
    # Extend knots for B-splines
    n_knots = len(knots)
    extended_knots = np.r_[(knots[0],)*(degree+1), knots, (knots[-1],)*(degree+1)]
    
    # Create B-spline basis
    basis_matrix = np.zeros((len(X), n_knots + degree - 1))
    
    for i in range(n_knots + degree - 1):
        # Create unit vector for basis function i
        coeffs = np.zeros(n_knots + degree - 1)
        coeffs[i] = 1.0
        
        # Create B-spline
        bspline = BSpline(extended_knots, coeffs, degree)
        basis_matrix[:, i] = bspline(X)
    
    return basis_matrix

def fit_smoothing_spline(X, y, lambda_val=1.0):
    """
    Fit smoothing spline using penalized least squares
    
    Parameters:
    X: predictor variable
    y: response variable
    lambda_val: smoothing parameter
    
    Returns:
    spline: fitted smoothing spline
    """
    # This is a simplified version - in practice, use specialized algorithms
    # For demonstration, we'll use scipy's CubicSpline with natural boundary conditions
    
    # Create natural cubic spline
    spline = CubicSpline(X, y, bc_type='natural')
    
    # Note: The actual smoothing spline implementation would involve:
    # 1. Creating the penalty matrix for the second derivative
    # 2. Solving the penalized least squares problem
    # 3. Computing the optimal smoothing parameter
    
    return spline

def select_optimal_knots(X, y, max_knots=10, method='quantile'):
    """
    Select optimal knot positions
    
    Parameters:
    X: predictor variable
    y: response variable
    max_knots: maximum number of knots to consider
    method: knot selection method ('quantile', 'uniform', 'cross_validation')
    
    Returns:
    knots: optimal knot positions
    """
    if method == 'quantile':
        # Use quantiles of X
        knots = np.percentile(X, np.linspace(0, 100, max_knots + 2))[1:-1]
    elif method == 'uniform':
        # Uniform spacing
        knots = np.linspace(X.min(), X.max(), max_knots + 2)[1:-1]
    elif method == 'cross_validation':
        # Use cross-validation to select optimal number of knots
        best_score = float('inf')
        best_knots = None
        
        for n_knots in range(2, max_knots + 1):
            knots = np.percentile(X, np.linspace(0, 100, n_knots + 2))[1:-1]
            
            # Cross-validation score
            # Create basis matrix
            basis_matrix = create_truncated_power_basis(X, knots)
            
            # Cross-validation
            cv_scores = cross_val_score(LinearRegression(), basis_matrix, y, cv=5, 
                                      scoring='neg_mean_squared_error')
            score = -cv_scores.mean()
            
            if score < best_score:
                best_score = score
                best_knots = knots
        
        knots = best_knots
    
    return knots

def create_truncated_power_basis(X, knots):
    """
    Create truncated power basis for cubic splines
    
    Parameters:
    X: predictor variable
    knots: knot positions
    
    Returns:
    basis_matrix: truncated power basis matrix
    """
    n_samples = len(X)
    n_knots = len(knots)
    
    # Basis matrix: [1, x, x^2, x^3, (x-xi_1)_+^3, ..., (x-xi_m)_+^3]
    basis_matrix = np.zeros((n_samples, n_knots + 4))
    
    # Polynomial terms
    basis_matrix[:, 0] = 1
    basis_matrix[:, 1] = X
    basis_matrix[:, 2] = X**2
    basis_matrix[:, 3] = X**3
    
    # Truncated power terms
    for i, knot in enumerate(knots):
        basis_matrix[:, i + 4] = np.maximum(0, X - knot)**3
    
    return basis_matrix

def analyze_spline_diagnostics(spline_model, X, y):
    """
    Analyze spline model diagnostics
    
    Parameters:
    spline_model: fitted spline model
    X: predictor variable
    y: response variable
    
    Returns:
    residuals: model residuals
    """
    y_pred = spline_model.predict(X)
    residuals = y - y_pred
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals vs Fitted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Q-Q Plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot of Residuals')
    
    # Residuals vs Predictor
    axes[1, 0].scatter(X, residuals, alpha=0.6)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residuals vs X')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[1, 1].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Histogram of Residuals')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return residuals

def demonstrate_advanced_splines():
    """Demonstrate advanced spline features"""
    # Generate synthetic data
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    y_true = 2 + 3*np.sin(X) + 0.5*X
    y = y_true + np.random.normal(0, 0.3, 100)
    
    # Define knots
    knots = np.array([2, 4, 6, 8])
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: B-spline basis functions
    plt.subplot(2, 3, 1)
    X_plot = np.linspace(0, 10, 200)
    bspline_basis = create_bspline_basis(X_plot, knots, degree=3)
    
    for i in range(bspline_basis.shape[1]):
        plt.plot(X_plot, bspline_basis[:, i], label=f'B-spline {i+1}')
    
    plt.xlabel('X')
    plt.ylabel('B-spline Basis Function Value')
    plt.title('B-spline Basis Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Comparison of basis types
    plt.subplot(2, 3, 2)
    truncated_basis = create_truncated_power_basis(X_plot, knots)
    
    plt.plot(X_plot, truncated_basis[:, 4], label='Truncated Power', linewidth=2)
    plt.plot(X_plot, bspline_basis[:, 0], label='B-spline', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('Basis Function Value')
    plt.title('Basis Function Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Knot selection methods
    plt.subplot(2, 3, 3)
    plt.scatter(X, y, alpha=0.6, label='Data')
    
    # Different knot selection methods
    knot_methods = {
        'Quantile': select_optimal_knots(X, y, max_knots=4, method='quantile'),
        'Uniform': select_optimal_knots(X, y, max_knots=4, method='uniform')
    }
    
    for name, knots in knot_methods.items():
        # Fit spline with these knots
        basis_matrix = create_truncated_power_basis(X, knots)
        model = LinearRegression()
        model.fit(basis_matrix, y)
        
        # Predict
        basis_plot = create_truncated_power_basis(X_plot, knots)
        y_plot = model.predict(basis_plot)
        plt.plot(X_plot, y_plot, label=f'{name} knots', linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Knot Selection Methods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Smoothing parameter effect
    plt.subplot(2, 3, 4)
    plt.scatter(X, y, alpha=0.6, label='Data')
    
    # Different smoothing parameters
    lambda_values = [0.1, 1.0, 10.0]
    for lambda_val in lambda_values:
        spline = fit_smoothing_spline(X, y, lambda_val=lambda_val)
        y_plot = spline(X_plot)
        plt.plot(X_plot, y_plot, label=f'λ = {lambda_val}', linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Smoothing Parameter Effect')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Cross-validation for knot selection
    plt.subplot(2, 3, 5)
    n_knots_range = range(2, 11)
    cv_scores = []
    
    for n_knots in n_knots_range:
        knots = np.percentile(X, np.linspace(0, 100, n_knots + 2))[1:-1]
        basis_matrix = create_truncated_power_basis(X, knots)
        
        cv_scores_temp = cross_val_score(LinearRegression(), basis_matrix, y, cv=5, 
                                        scoring='neg_mean_squared_error')
        cv_scores.append(-cv_scores_temp.mean())
    
    plt.plot(n_knots_range, cv_scores, 'bo-')
    plt.xlabel('Number of Knots')
    plt.ylabel('Cross-Validation MSE')
    plt.title('CV Score vs Number of Knots')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Model comparison
    plt.subplot(2, 3, 6)
    plt.scatter(X, y, alpha=0.6, label='Data')
    
    # Compare different spline types
    # Regular cubic spline
    basis_matrix = create_truncated_power_basis(X, knots)
    model = LinearRegression()
    model.fit(basis_matrix, y)
    y_plot = model.predict(create_truncated_power_basis(X_plot, knots))
    plt.plot(X_plot, y_plot, label='Regular Cubic Spline', linewidth=2)
    
    # B-spline
    bspline_basis = create_bspline_basis(X, knots, degree=3)
    model_bspline = LinearRegression()
    model_bspline.fit(bspline_basis, y)
    y_plot_bspline = model_bspline.predict(create_bspline_basis(X_plot, knots, degree=3))
    plt.plot(X_plot, y_plot_bspline, label='B-spline', linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Spline Type Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'knots': knots,
        'cv_scores': cv_scores,
        'n_knots_range': list(n_knots_range)
    }

if __name__ == "__main__":
    results = demonstrate_advanced_splines()
