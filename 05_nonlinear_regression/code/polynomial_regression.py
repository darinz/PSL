"""
Polynomial Regression Implementation
===================================

This module provides a complete implementation of polynomial regression
including model fitting, prediction, evaluation, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import pandas as pd
from .polynomial_utilities import create_orthogonal_polynomials

class PolynomialRegression:
    def __init__(self, degree=2, use_orthogonal=False):
        """
        Polynomial Regression Model
        
        Parameters:
        degree: polynomial degree
        use_orthogonal: whether to use orthogonal polynomials
        """
        self.degree = degree
        self.use_orthogonal = use_orthogonal
        self.model = LinearRegression()
        self.poly_features = None
        self.X_poly = None
        self.coefficients = None
        self.intercept = None
        
    def fit(self, X, y):
        """Fit polynomial regression model"""
        if self.use_orthogonal:
            self.X_poly = create_orthogonal_polynomials(X, self.degree)
        else:
            self.poly_features = PolynomialFeatures(degree=self.degree, include_bias=True)
            self.X_poly = self.poly_features.fit_transform(X.reshape(-1, 1))
        
        # Fit linear regression
        self.model.fit(self.X_poly, y)
        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.use_orthogonal:
            X_poly = create_orthogonal_polynomials(X, self.degree)
        else:
            X_poly = self.poly_features.transform(X.reshape(-1, 1))
        
        return self.model.predict(X_poly)
    
    def get_polynomial_equation(self):
        """Get the polynomial equation as a string"""
        if self.use_orthogonal:
            return "Orthogonal polynomial coefficients: " + str(self.coefficients)
        
        equation = f"y = {self.intercept:.4f}"
        for i, coef in enumerate(self.coefficients[1:], 1):
            if coef >= 0:
                equation += f" + {coef:.4f}x^{i}"
            else:
                equation += f" - {abs(coef):.4f}x^{i}"
        
        return equation
    
    def calculate_metrics(self, X, y):
        """Calculate model performance metrics"""
        y_pred = self.predict(X)
        
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        # Adjusted R-squared
        n = len(y)
        p = self.degree + 1
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        # AIC and BIC
        rss = np.sum((y - y_pred)**2)
        aic = n * np.log(rss/n) + 2 * p
        bic = n * np.log(rss/n) + p * np.log(n)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2,
            'Adjusted R²': adj_r2,
            'AIC': aic,
            'BIC': bic
        }

def demonstrate_polynomial_regression():
    """Demonstrate polynomial regression with synthetic data"""
    # Generate synthetic data
    np.random.seed(42)
    X = np.linspace(-3, 3, 100)
    y_true = 2 + 3*X - 0.5*X**2 + 0.1*X**3
    y = y_true + np.random.normal(0, 0.5, 100)
    
    # Test different polynomial degrees
    degrees = [1, 2, 3, 4, 5, 6]
    models = {}
    metrics = {}
    
    for degree in degrees:
        # Fit model
        model = PolynomialRegression(degree=degree)
        model.fit(X, y)
        models[degree] = model
        
        # Calculate metrics
        metrics[degree] = model.calculate_metrics(X, y)
        
        print(f"Degree {degree}:")
        print(f"  Equation: {model.get_polynomial_equation()}")
        print(f"  R²: {metrics[degree]['R²']:.4f}")
        print(f"  AIC: {metrics[degree]['AIC']:.4f}")
        print(f"  BIC: {metrics[degree]['BIC']:.4f}")
        print()
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Data and fitted curves
    plt.subplot(2, 3, 1)
    plt.scatter(X, y, alpha=0.6, label='Data')
    X_plot = np.linspace(-3, 3, 200)
    
    for degree in [1, 2, 3]:
        y_plot = models[degree].predict(X_plot)
        plt.plot(X_plot, y_plot, label=f'Degree {degree}')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Polynomial Fits')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: R² vs Degree
    plt.subplot(2, 3, 2)
    r2_values = [metrics[d]['R²'] for d in degrees]
    plt.plot(degrees, r2_values, 'bo-')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('R²')
    plt.title('R² vs Degree')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: AIC vs Degree
    plt.subplot(2, 3, 3)
    aic_values = [metrics[d]['AIC'] for d in degrees]
    plt.plot(degrees, aic_values, 'ro-')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('AIC')
    plt.title('AIC vs Degree')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: BIC vs Degree
    plt.subplot(2, 3, 4)
    bic_values = [metrics[d]['BIC'] for d in degrees]
    plt.plot(degrees, bic_values, 'go-')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('BIC')
    plt.title('BIC vs Degree')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Residuals for degree 3
    plt.subplot(2, 3, 5)
    y_pred_3 = models[3].predict(X)
    residuals_3 = y - y_pred_3
    plt.scatter(y_pred_3, residuals_3, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals (Degree 3)')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Overfitting demonstration
    plt.subplot(2, 3, 6)
    plt.scatter(X, y, alpha=0.6, label='Data')
    
    for degree in [3, 6]:
        y_plot = models[degree].predict(X_plot)
        plt.plot(X_plot, y_plot, label=f'Degree {degree}')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Overfitting Example')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return models, metrics

def analyze_polynomial_residuals(model, X, y):
    """
    Analyze residuals for polynomial regression
    """
    y_pred = model.predict(X)
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
    
    # Statistical tests
    from scipy.stats import shapiro, jarque_bera
    
    # Shapiro-Wilk test for normality
    shapiro_stat, shapiro_p = shapiro(residuals)
    print(f"Shapiro-Wilk test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
    
    # Jarque-Bera test for normality
    jb_stat, jb_p = jarque_bera(residuals)
    print(f"Jarque-Bera test: statistic={jb_stat:.4f}, p-value={jb_p:.4f}")
    
    return residuals

def cross_validate_polynomial_degree(X, y, max_degree=10, cv_folds=5):
    """
    Cross-validation for polynomial degree selection
    """
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []
    
    for degree in range(1, max_degree + 1):
        fold_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Fit model
            model = PolynomialRegression(degree=degree)
            model.fit(X_train, y_train)
            
            # Predict and calculate MSE
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            fold_scores.append(mse)
        
        cv_scores.append(np.mean(fold_scores))
    
    # Find optimal degree
    optimal_degree = np.argmin(cv_scores) + 1
    
    # Plot CV scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_degree + 1), cv_scores, 'bo-')
    plt.axvline(x=optimal_degree, color='r', linestyle='--', 
                label=f'Optimal degree: {optimal_degree}')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Cross-Validation MSE')
    plt.title('Cross-Validation for Polynomial Degree Selection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return optimal_degree, cv_scores

if __name__ == "__main__":
    # Demonstrate polynomial regression
    models, metrics = demonstrate_polynomial_regression()
    
    # Demonstrate cross-validation
    np.random.seed(42)
    X = np.linspace(-3, 3, 100)
    y_true = 2 + 3*X - 0.5*X**2 + 0.1*X**3
    y = y_true + np.random.normal(0, 0.5, 100)
    
    optimal_degree, cv_scores = cross_validate_polynomial_degree(X, y, max_degree=8)
    print(f"Optimal degree from CV: {optimal_degree}")
    
    # Demonstrate residual analysis
    best_model = PolynomialRegression(degree=optimal_degree)
    best_model.fit(X, y)
    residuals = analyze_polynomial_residuals(best_model, X, y)
