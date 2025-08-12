"""
Smoothing Spline Regression Implementation
=========================================

This module provides a complete implementation of smoothing splines
including the SmoothingSpline class, cross-validation, and comprehensive demonstrations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.linalg import solve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import pandas as pd
from scipy import stats

class SmoothingSpline:
    def __init__(self, lambda_val=None, df=None, cv=True):
        """
        Smoothing Spline Implementation
        
        Parameters:
        lambda_val: smoothing parameter
        df: effective degrees of freedom
        cv: whether to use cross-validation for lambda selection
        """
        self.lambda_val = lambda_val
        self.df = df
        self.cv = cv
        self.X = None
        self.y = None
        self.beta = None
        self.smoother_matrix = None
        self.edf = None
        
    def create_natural_spline_basis(self, X):
        """
        Create natural cubic spline basis matrix
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
    
    def create_penalty_matrix(self, X):
        """
        Create penalty matrix for natural cubic splines
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
    
    def fit(self, X, y):
        """Fit smoothing spline"""
        # Sort data by X
        sort_idx = np.argsort(X)
        self.X = X[sort_idx]
        self.y = y[sort_idx]
        
        n = len(self.X)
        
        # Create basis matrix and penalty matrix
        H = self.create_natural_spline_basis(self.X)
        Omega = self.create_penalty_matrix(self.X)
        
        # Select lambda if not provided
        if self.lambda_val is None:
            if self.df is not None:
                # Find lambda that gives desired degrees of freedom
                self.lambda_val = self.find_lambda_for_df(H, Omega, self.df)
            elif self.cv:
                # Use cross-validation to select lambda
                self.lambda_val = self.select_lambda_cv(H, Omega)
            else:
                # Default lambda
                self.lambda_val = 1.0
        
        # Solve for coefficients
        self.beta = solve(H.T @ H + self.lambda_val * Omega, H.T @ self.y)
        
        # Compute smoother matrix
        self.smoother_matrix = H @ solve(H.T @ H + self.lambda_val * Omega, H.T)
        
        # Compute effective degrees of freedom
        self.edf = np.trace(self.smoother_matrix)
        
        return self
    
    def find_lambda_for_df(self, H, Omega, target_df):
        """Find lambda that gives desired degrees of freedom"""
        def objective(lambda_val):
            S = H @ solve(H.T @ H + lambda_val * Omega, H.T)
            edf = np.trace(S)
            return (edf - target_df)**2
        
        # Use binary search to find optimal lambda
        lambda_min, lambda_max = 1e-6, 1e6
        for _ in range(20):
            lambda_mid = np.sqrt(lambda_min * lambda_max)
            if objective(lambda_mid) < 1e-6:
                break
            if np.trace(H @ solve(H.T @ H + lambda_mid * Omega, H.T)) > target_df:
                lambda_min = lambda_mid
            else:
                lambda_max = lambda_mid
        
        return lambda_mid
    
    def select_lambda_cv(self, H, Omega):
        """Select lambda using cross-validation"""
        lambda_candidates = np.logspace(-3, 3, 20)
        cv_scores = []
        
        for lambda_val in lambda_candidates:
            S = H @ solve(H.T @ H + lambda_val * Omega, H.T)
            # Leave-one-out cross-validation
            y_pred = S @ self.y
            residuals = self.y - y_pred
            # Adjust for leverage
            leverage = np.diag(S)
            adjusted_residuals = residuals / (1 - leverage)
            cv_score = np.mean(adjusted_residuals**2)
            cv_scores.append(cv_score)
        
        return lambda_candidates[np.argmin(cv_scores)]
    
    def predict(self, X_new):
        """Make predictions"""
        if self.beta is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Create basis matrix for new points
        H_new = self.create_natural_spline_basis(X_new)
        
        # Predict using fitted coefficients
        return H_new @ self.beta
    
    def get_spline_function(self):
        """Get the fitted spline function"""
        if self.beta is None:
            raise ValueError("Model must be fitted before getting spline function")
        
        def spline_func(x):
            return self.predict(x)
        return spline_func

def demonstrate_smoothing_splines():
    """Demonstrate smoothing spline fitting"""
    # Generate synthetic data
    np.random.seed(42)
    X = np.linspace(0, 10, 50)
    y_true = 2 + 3*np.sin(X) + 0.5*X
    y = y_true + np.random.normal(0, 0.5, 50)
    
    # Test different lambda values
    lambda_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    models = {}
    
    for lambda_val in lambda_values:
        model = SmoothingSpline(lambda_val=lambda_val, cv=False)
        model.fit(X, y)
        models[f'λ={lambda_val}'] = model
    
    # Test different degrees of freedom
    df_values = [3, 5, 8, 12, 20, 30]
    models_df = {}
    
    for df in df_values:
        model = SmoothingSpline(df=df, cv=False)
        model.fit(X, y)
        models_df[f'DF={df}'] = model
    
    # Evaluate models
    X_plot = np.linspace(0, 10, 200)
    
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Effect of lambda
    plt.subplot(3, 2, 1)
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.plot(X, y_true, 'k--', label='True Function', linewidth=2)
    
    for name, model in models.items():
        y_plot = model.predict(X_plot)
        plt.plot(X_plot, y_plot, label=name, linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Effect of Smoothing Parameter λ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Effect of degrees of freedom
    plt.subplot(3, 2, 2)
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.plot(X, y_true, 'k--', label='True Function', linewidth=2)
    
    for name, model in models_df.items():
        y_plot = model.predict(X_plot)
        plt.plot(X_plot, y_plot, label=name, linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Effect of Degrees of Freedom')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Lambda vs EDF
    plt.subplot(3, 2, 3)
    lambda_list = []
    edf_list = []
    
    for name, model in models.items():
        lambda_val = float(name.split('=')[1])
        lambda_list.append(lambda_val)
        edf_list.append(model.edf)
    
    plt.semilogx(lambda_list, edf_list, 'bo-')
    plt.xlabel('λ')
    plt.ylabel('Effective Degrees of Freedom')
    plt.title('λ vs Effective Degrees of Freedom')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Cross-validation
    plt.subplot(3, 2, 4)
    lambda_candidates = np.logspace(-3, 3, 20)
    cv_scores = []
    
    for lambda_val in lambda_candidates:
        model = SmoothingSpline(lambda_val=lambda_val, cv=False)
        model.fit(X, y)
        
        # Compute LOOCV score
        y_pred = model.smoother_matrix @ y
        residuals = y - y_pred
        leverage = np.diag(model.smoother_matrix)
        adjusted_residuals = residuals / (1 - leverage)
        cv_score = np.mean(adjusted_residuals**2)
        cv_scores.append(cv_score)
    
    plt.semilogx(lambda_candidates, cv_scores, 'ro-')
    plt.xlabel('λ')
    plt.ylabel('LOOCV Score')
    plt.title('Cross-Validation for λ Selection')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Residuals
    plt.subplot(3, 2, 5)
    best_model = models['λ=1.0']  # Choose a reasonable model
    y_pred = best_model.smoother_matrix @ y
    residuals = y - y_pred
    
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Smoother matrix
    plt.subplot(3, 2, 6)
    plt.imshow(best_model.smoother_matrix, cmap='viridis')
    plt.colorbar()
    plt.title('Smoother Matrix S_λ')
    plt.xlabel('j')
    plt.ylabel('i')
    
    plt.tight_layout()
    plt.show()
    
    return models, models_df

def analyze_noisy_data():
    """Analyze smoothing splines on noisy data"""
    # Generate noisy data with different noise levels
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    y_true = 2 + 3*np.sin(X) + 0.5*X
    
    noise_levels = [0.1, 0.5, 1.0, 2.0]
    models = {}
    
    for noise in noise_levels:
        y = y_true + np.random.normal(0, noise, 100)
        
        # Fit smoothing spline with cross-validation
        model = SmoothingSpline(cv=True)
        model.fit(X, y)
        models[f'Noise={noise}'] = model
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.plot(X, y_true, 'k--', label='True Function', linewidth=2)
    
    for name, model in models.items():
        y_plot = model.predict(X)
        plt.plot(X, y_plot, label=f'{name}, λ={model.lambda_val:.3f}', linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Smoothing Splines on Noisy Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    noise_list = []
    lambda_list = []
    edf_list = []
    
    for name, model in models.items():
        noise = float(name.split('=')[1])
        noise_list.append(noise)
        lambda_list.append(model.lambda_val)
        edf_list.append(model.edf)
    
    plt.plot(noise_list, lambda_list, 'bo-', label='λ')
    plt.xlabel('Noise Level')
    plt.ylabel('Selected λ')
    plt.title('λ Selection vs Noise Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(noise_list, edf_list, 'ro-', label='EDF')
    plt.xlabel('Noise Level')
    plt.ylabel('Effective Degrees of Freedom')
    plt.title('EDF vs Noise Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Show smoother matrix for one model
    model = models['Noise=0.5']
    plt.imshow(model.smoother_matrix, cmap='viridis')
    plt.colorbar()
    plt.title('Smoother Matrix (Noise=0.5)')
    plt.xlabel('j')
    plt.ylabel('i')
    
    plt.tight_layout()
    plt.show()
    
    return models

if __name__ == "__main__":
    print("Demonstrating Smoothing Splines...")
    models, models_df = demonstrate_smoothing_splines()
    
    print("\nAnalyzing Noisy Data...")
    noisy_models = analyze_noisy_data()
