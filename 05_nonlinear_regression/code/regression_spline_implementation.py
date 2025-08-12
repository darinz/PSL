"""
Regression Spline Implementation
===============================

This module provides a complete implementation of regression splines
including the RegressionSpline class, cross-validation, and comprehensive demonstrations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from scipy import stats

class RegressionSpline:
    def __init__(self, df=None, knots=None, spline_type='cubic', 
                 regularization=None, lambda_val=1.0):
        """
        Regression Spline Implementation
        
        Parameters:
        df: degrees of freedom (number of basis functions)
        knots: array of knot positions
        spline_type: 'cubic' or 'natural'
        regularization: 'ridge', 'lasso', or None
        lambda_val: regularization parameter
        """
        self.df = df
        self.knots = knots
        self.spline_type = spline_type
        self.regularization = regularization
        self.lambda_val = lambda_val
        self.model = None
        self.basis_matrix = None
        self.coefficients = None
        self.intercept = None
        
    def create_basis_matrix(self, X):
        """
        Create basis matrix for regression splines
        """
        n_samples = len(X)
        
        if self.spline_type == 'cubic':
            # Cubic spline basis
            if self.knots is not None:
                n_knots = len(self.knots)
                n_basis = n_knots + 4
            else:
                # Use df to determine number of knots
                n_knots = self.df - 4
                self.knots = np.percentile(X, np.linspace(0, 100, n_knots + 2))[1:-1]
                n_basis = self.df
        else:
            # Natural cubic spline basis
            if self.knots is not None:
                n_knots = len(self.knots)
                n_basis = n_knots
            else:
                n_knots = self.df
                self.knots = np.percentile(X, np.linspace(0, 100, n_knots + 2))[1:-1]
                n_basis = self.df
        
        basis_matrix = np.zeros((n_samples, n_basis))
        
        if self.spline_type == 'cubic':
            # Polynomial terms
            basis_matrix[:, 0] = 1
            basis_matrix[:, 1] = X
            basis_matrix[:, 2] = X**2
            basis_matrix[:, 3] = X**3
            
            # Truncated power terms
            for i, knot in enumerate(self.knots):
                basis_matrix[:, i + 4] = np.maximum(0, X - knot)**3
        else:
            # Natural cubic spline basis using scipy
            for i in range(n_knots):
                # Create unit vector for basis function i
                unit_vector = np.zeros(n_knots)
                unit_vector[i] = 1.0
                
                # Create natural cubic spline
                temp_spline = CubicSpline(self.knots, unit_vector, bc_type='natural')
                basis_matrix[:, i] = temp_spline(X)
        
        return basis_matrix
    
    def fit(self, X, y):
        """Fit regression spline model"""
        self.basis_matrix = self.create_basis_matrix(X)
        
        if self.regularization == 'ridge':
            self.model = Ridge(alpha=self.lambda_val)
        elif self.regularization == 'lasso':
            self.model = Lasso(alpha=self.lambda_val)
        else:
            self.model = LinearRegression()
        
        self.model.fit(self.basis_matrix, y)
        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        basis_matrix = self.create_basis_matrix(X)
        return self.intercept + basis_matrix @ self.coefficients
    
    def get_spline_function(self):
        """Get the fitted spline function"""
        def spline_func(x):
            basis = self.create_basis_matrix(x)
            return self.intercept + basis @ self.coefficients
        return spline_func

def demonstrate_regression_splines():
    """Demonstrate regression spline fitting"""
    # Generate synthetic data
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    y_true = 2 + 3*np.sin(X) + 0.5*X
    y = y_true + np.random.normal(0, 0.3, 100)
    
    # Test different degrees of freedom
    df_values = [4, 6, 8, 10, 12, 15]
    models = {}
    
    for df in df_values:
        model = RegressionSpline(df=df, spline_type='cubic')
        model.fit(X, y)
        models[f'DF={df}'] = model
    
    # Evaluate models
    X_plot = np.linspace(0, 10, 200)
    
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Spline fits
    plt.subplot(3, 2, 1)
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.plot(X, y_true, 'k--', label='True Function', linewidth=2)
    
    for name, model in models.items():
        y_plot = model.predict(X_plot)
        plt.plot(X_plot, y_plot, label=name, linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Regression Spline Fits')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Model comparison
    plt.subplot(3, 2, 2)
    df_list = []
    mse_list = []
    r2_list = []
    
    for name, model in models.items():
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        df_val = int(name.split('=')[1])
        df_list.append(df_val)
        mse_list.append(mse)
        r2_list.append(r2)
    
    plt.plot(df_list, mse_list, 'bo-', label='MSE')
    plt.xlabel('Degrees of Freedom')
    plt.ylabel('Mean Squared Error')
    plt.title('Model Performance vs DF')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Cross-validation
    plt.subplot(3, 2, 3)
    cv_scores = []
    
    for df in df_values:
        model = RegressionSpline(df=df, spline_type='cubic')
        cv_score = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        cv_scores.append(-cv_score.mean())
    
    plt.plot(df_values, cv_scores, 'ro-', label='CV MSE')
    plt.xlabel('Degrees of Freedom')
    plt.ylabel('Cross-Validation MSE')
    plt.title('Cross-Validation Performance')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Basis functions
    plt.subplot(3, 2, 4)
    model = models['DF=8']
    basis_matrix = model.create_basis_matrix(X_plot)
    
    for i in range(basis_matrix.shape[1]):
        plt.plot(X_plot, basis_matrix[:, i], label=f'Basis {i+1}')
    
    plt.xlabel('X')
    plt.ylabel('Basis Function Value')
    plt.title('Basis Functions (DF=8)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Residuals
    plt.subplot(3, 2, 5)
    best_model = models['DF=8']  # Choose a reasonable model
    y_pred = best_model.predict(X)
    residuals = y - y_pred
    
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Regularization comparison
    plt.subplot(3, 2, 6)
    lambda_values = [0.01, 0.1, 1.0, 10.0]
    
    for lambda_val in lambda_values:
        model_ridge = RegressionSpline(df=12, regularization='ridge', lambda_val=lambda_val)
        model_ridge.fit(X, y)
        y_plot = model_ridge.predict(X_plot)
        plt.plot(X_plot, y_plot, label=f'λ={lambda_val}')
    
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Ridge Regularization Effect')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return models

def analyze_birthrate_data():
    """Analyze birthrate data with regression splines"""
    # Generate birthrate-like data (simulated)
    np.random.seed(42)
    years = np.arange(1960, 2020)
    # Simulate birthrate with some trend and noise
    birthrate = 20 - 0.1*(years - 1960) + 2*np.sin(2*np.pi*(years - 1960)/20) + np.random.normal(0, 0.5, len(years))
    
    # Test different degrees of freedom
    df_values = [3, 5, 7, 10, 15, 20]
    models = {}
    
    for df in df_values:
        model = RegressionSpline(df=df, spline_type='natural')
        model.fit(years, birthrate)
        models[f'DF={df}'] = model
    
    # Cross-validation to select optimal df
    cv_scores = []
    for df in df_values:
        model = RegressionSpline(df=df, spline_type='natural')
        cv_score = cross_val_score(model, years, birthrate, cv=5, scoring='neg_mean_squared_error')
        cv_scores.append(-cv_score.mean())
    
    optimal_df = df_values[np.argmin(cv_scores)]
    print(f"Optimal degrees of freedom: {optimal_df}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(years, birthrate, alpha=0.7, label='Data')
    
    years_plot = np.linspace(1960, 2020, 200)
    for name, model in models.items():
        y_plot = model.predict(years_plot)
        plt.plot(years_plot, y_plot, label=name, linewidth=2)
    
    plt.xlabel('Year')
    plt.ylabel('Birthrate')
    plt.title('Birthrate Data: Spline Fits')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(df_values, cv_scores, 'bo-')
    plt.axvline(x=optimal_df, color='r', linestyle='--', label=f'Optimal DF={optimal_df}')
    plt.xlabel('Degrees of Freedom')
    plt.ylabel('Cross-Validation MSE')
    plt.title('Model Selection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    best_model = models[f'DF={optimal_df}']
    y_pred = best_model.predict(years)
    residuals = birthrate - y_pred
    
    plt.scatter(years, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Year')
    plt.ylabel('Residuals')
    plt.title('Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    
    plt.tight_layout()
    plt.show()
    
    return models, optimal_df

if __name__ == "__main__":
    print("Demonstrating Regression Splines...")
    models = demonstrate_regression_splines()
    
    print("\nAnalyzing Birthrate Data...")
    birthrate_models, optimal_df = analyze_birthrate_data()
