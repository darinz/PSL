"""
Cubic Spline Regression Implementation
=====================================

This module provides a complete implementation of cubic spline regression
including regular and natural cubic splines, basis functions, and comprehensive demonstrations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, BSpline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class CubicSplineRegression:
    def __init__(self, knots=None, natural=False):
        """
        Cubic Spline Regression
        
        Parameters:
        knots: array of knot positions
        natural: whether to use natural cubic splines
        """
        self.knots = knots
        self.natural = natural
        self.spline = None
        self.basis_matrix = None
        self.coefficients = None
        
    def create_truncated_power_basis(self, X):
        """
        Create truncated power basis for cubic splines
        """
        n_samples = len(X)
        n_knots = len(self.knots)
        
        # Basis matrix: [1, x, x^2, x^3, (x-xi_1)_+^3, ..., (x-xi_m)_+^3]
        basis_matrix = np.zeros((n_samples, n_knots + 4))
        
        # Polynomial terms
        basis_matrix[:, 0] = 1
        basis_matrix[:, 1] = X
        basis_matrix[:, 2] = X**2
        basis_matrix[:, 3] = X**3
        
        # Truncated power terms
        for i, knot in enumerate(self.knots):
            basis_matrix[:, i + 4] = np.maximum(0, X - knot)**3
        
        return basis_matrix
    
    def create_natural_spline_basis(self, X):
        """
        Create natural cubic spline basis
        """
        n_samples = len(X)
        n_knots = len(self.knots)
        
        # For natural splines, we need to construct the basis differently
        # This is a simplified version - in practice, use specialized libraries
        basis_matrix = np.zeros((n_samples, n_knots))
        
        # Use scipy's natural cubic spline
        self.spline = CubicSpline(self.knots, np.zeros(n_knots), bc_type='natural')
        
        # Create basis functions by evaluating at different points
        for i in range(n_knots):
            # Create a unit vector at knot i
            unit_vector = np.zeros(n_knots)
            unit_vector[i] = 1.0
            
            # Create spline with this unit vector
            temp_spline = CubicSpline(self.knots, unit_vector, bc_type='natural')
            basis_matrix[:, i] = temp_spline(X)
        
        return basis_matrix
    
    def fit(self, X, y):
        """Fit cubic spline regression"""
        if self.knots is None:
            # Use quantiles of X as knots
            self.knots = np.percentile(X, np.linspace(0, 100, 6))[1:-1]
        
        if self.natural:
            self.basis_matrix = self.create_natural_spline_basis(X)
        else:
            self.basis_matrix = self.create_truncated_power_basis(X)
        
        # Fit linear regression on basis functions
        model = LinearRegression()
        model.fit(self.basis_matrix, y)
        self.coefficients = model.coef_
        self.intercept = model.intercept_
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.natural:
            basis_matrix = self.create_natural_spline_basis(X)
        else:
            basis_matrix = self.create_truncated_power_basis(X)
        
        return self.intercept + basis_matrix @ self.coefficients
    
    def get_spline_function(self):
        """Get the fitted spline function"""
        if self.natural:
            # For natural splines, use scipy's CubicSpline
            return self.spline
        else:
            # For regular cubic splines, create a function
            def spline_func(x):
                basis = self.create_truncated_power_basis(x)
                return self.intercept + basis @ self.coefficients
            return spline_func

def demonstrate_cubic_splines():
    """Demonstrate cubic spline regression"""
    # Generate synthetic data with nonlinear relationship
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    y_true = 2 + 3*np.sin(X) + 0.5*X
    y = y_true + np.random.normal(0, 0.3, 100)
    
    # Define knots
    knots = np.array([2, 4, 6, 8])
    
    # Fit different types of splines
    splines = {}
    
    # Regular cubic spline
    spline_reg = CubicSplineRegression(knots=knots, natural=False)
    spline_reg.fit(X, y)
    splines['Regular'] = spline_reg
    
    # Natural cubic spline
    spline_nat = CubicSplineRegression(knots=knots, natural=True)
    spline_nat.fit(X, y)
    splines['Natural'] = spline_nat
    
    # Scipy cubic spline for comparison
    from scipy.interpolate import CubicSpline
    scipy_spline = CubicSpline(X, y, bc_type='natural')
    splines['Scipy'] = scipy_spline
    
    # Evaluate and plot
    X_plot = np.linspace(0, 10, 200)
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Data and spline fits
    plt.subplot(2, 3, 1)
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.plot(X, y_true, 'k--', label='True Function', linewidth=2)
    
    for name, spline in splines.items():
        if name == 'Scipy':
            y_plot = spline(X_plot)
        else:
            y_plot = spline.predict(X_plot)
        plt.plot(X_plot, y_plot, label=f'{name} Spline', linewidth=2)
    
    # Mark knots
    for knot in knots:
        plt.axvline(x=knot, color='gray', linestyle=':', alpha=0.7)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Cubic Spline Fits')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Basis functions for regular spline
    plt.subplot(2, 3, 2)
    basis_matrix = spline_reg.create_truncated_power_basis(X_plot)
    
    for i in range(basis_matrix.shape[1]):
        plt.plot(X_plot, basis_matrix[:, i], label=f'Basis {i+1}')
    
    plt.xlabel('X')
    plt.ylabel('Basis Function Value')
    plt.title('Truncated Power Basis Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: First derivatives
    plt.subplot(2, 3, 3)
    for name, spline in splines.items():
        if name == 'Scipy':
            y_deriv = spline.derivative()(X_plot)
        else:
            # Numerical derivative
            h = 1e-6
            y_plot_plus = spline.predict(X_plot + h)
            y_plot_minus = spline.predict(X_plot - h)
            y_deriv = (y_plot_plus - y_plot_minus) / (2*h)
        
        plt.plot(X_plot, y_deriv, label=f'{name} Spline')
    
    plt.xlabel('X')
    plt.ylabel('First Derivative')
    plt.title('First Derivatives')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Second derivatives
    plt.subplot(2, 3, 4)
    for name, spline in splines.items():
        if name == 'Scipy':
            y_deriv2 = spline.derivative(2)(X_plot)
        else:
            # Numerical second derivative
            h = 1e-6
            y_plot_plus = spline.predict(X_plot + h)
            y_plot_minus = spline.predict(X_plot - h)
            y_plot = spline.predict(X_plot)
            y_deriv2 = (y_plot_plus - 2*y_plot + y_plot_minus) / h**2
        
        plt.plot(X_plot, y_deriv2, label=f'{name} Spline')
    
    plt.xlabel('X')
    plt.ylabel('Second Derivative')
    plt.title('Second Derivatives')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Residuals
    plt.subplot(2, 3, 5)
    for name, spline in splines.items():
        if name == 'Scipy':
            y_pred = spline(X)
        else:
            y_pred = spline.predict(X)
        
        residuals = y - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6, label=f'{name} Spline')
    
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Knot placement effect
    plt.subplot(2, 3, 6)
    plt.scatter(X, y, alpha=0.6, label='Data')
    
    # Different knot placements
    knot_configs = {
        'Few knots': np.array([3, 7]),
        'Many knots': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        'Optimal knots': np.array([2, 4, 6, 8])
    }
    
    for name, knots in knot_configs.items():
        spline = CubicSplineRegression(knots=knots, natural=True)
        spline.fit(X, y)
        y_plot = spline.predict(X_plot)
        plt.plot(X_plot, y_plot, label=name, linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Effect of Knot Placement')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return splines

if __name__ == "__main__":
    spline_models = demonstrate_cubic_splines()
