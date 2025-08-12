"""
Local Regression Implementation
==============================

This module provides a complete implementation of local regression
including the LocalRegression class, cross-validation, and comprehensive demonstrations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
import pandas as pd
from scipy import stats

class LocalRegression:
    def __init__(self, degree=1, bandwidth=None, kernel='tricube', 
                 nn_frac=0.3, robust=False, iterations=3):
        """
        Local Regression Implementation
        
        Parameters:
        degree: polynomial degree for local fits
        bandwidth: fixed bandwidth (if None, use nearest neighbor)
        kernel: kernel function ('tricube', 'gaussian', 'epanechnikov')
        nn_frac: fraction of points for nearest neighbor bandwidth
        robust: whether to use robust fitting (LOWESS)
        iterations: number of iterations for robust fitting
        """
        self.degree = degree
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.nn_frac = nn_frac
        self.robust = robust
        self.iterations = iterations
        self.X = None
        self.y = None
        
    def kernel_function(self, u):
        """Compute kernel weights"""
        if self.kernel == 'tricube':
            return np.where(np.abs(u) < 1, (1 - np.abs(u)**3)**3, 0)
        elif self.kernel == 'gaussian':
            return np.exp(-u**2 / 2)
        elif self.kernel == 'epanechnikov':
            return np.where(np.abs(u) < 1, 0.75 * (1 - u**2), 0)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def compute_bandwidth(self, X, x0):
        """Compute bandwidth for prediction point x0"""
        if self.bandwidth is not None:
            return self.bandwidth
        
        # Nearest neighbor bandwidth
        n_neighbors = max(1, int(self.nn_frac * len(X)))
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(X.reshape(-1, 1))
        
        distances, _ = nn.kneighbors([[x0]])
        return distances[0, -1]
    
    def local_fit(self, X, y, x0, bandwidth):
        """Fit local polynomial at point x0"""
        # Compute distances and weights
        distances = np.abs(X - x0)
        u = distances / bandwidth
        weights = self.kernel_function(u)
        
        # Remove points with zero weight
        mask = weights > 0
        if np.sum(mask) < self.degree + 1:
            return np.nan
        
        X_local = X[mask]
        y_local = y[mask]
        weights_local = weights[mask]
        
        # Create polynomial basis
        X_poly = np.ones((len(X_local), self.degree + 1))
        for d in range(1, self.degree + 1):
            X_poly[:, d] = (X_local - x0)**d
        
        # Weighted least squares
        W = np.diag(weights_local)
        XWX = X_poly.T @ W @ X_poly
        XWy = X_poly.T @ W @ y_local
        
        try:
            beta = np.linalg.solve(XWX, XWy)
            return beta[0]  # Return intercept (prediction at x0)
        except np.linalg.LinAlgError:
            return np.nan
    
    def robust_weights(self, residuals):
        """Compute robust weights for LOWESS"""
        # Bisquare weight function
        u = residuals / (6 * np.median(np.abs(residuals)))
        return np.where(np.abs(u) < 1, (1 - u**2)**2, 0)
    
    def fit(self, X, y):
        """Fit local regression model"""
        self.X = np.array(X)
        self.y = np.array(y)
        return self
    
    def predict(self, X_new):
        """Make predictions"""
        if self.X is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X_new = np.array(X_new)
        predictions = np.zeros(len(X_new))
        
        for i, x0 in enumerate(X_new):
            if self.robust:
                # Robust fitting (LOWESS)
                pred = self._robust_predict(x0)
            else:
                # Standard local regression
                bandwidth = self.compute_bandwidth(self.X, x0)
                pred = self.local_fit(self.X, self.y, x0, bandwidth)
            
            predictions[i] = pred
        
        return predictions
    
    def _robust_predict(self, x0):
        """Robust prediction using LOWESS algorithm"""
        # Initial fit
        bandwidth = self.compute_bandwidth(self.X, x0)
        pred = self.local_fit(self.X, self.y, x0, bandwidth)
        
        if np.isnan(pred):
            return np.nan
        
        # Iterative robust fitting
        for _ in range(self.iterations):
            # Compute residuals
            all_preds = np.array([self.local_fit(self.X, self.y, xi, bandwidth) 
                                 for xi in self.X])
            residuals = self.y - all_preds
            
            # Compute robust weights
            robust_weights = self.robust_weights(residuals)
            
            # Refit with robust weights
            distances = np.abs(self.X - x0)
            u = distances / bandwidth
            kernel_weights = self.kernel_function(u)
            combined_weights = kernel_weights * robust_weights
            
            # Weighted local fit
            mask = combined_weights > 0
            if np.sum(mask) < self.degree + 1:
                break
            
            X_local = self.X[mask]
            y_local = self.y[mask]
            weights_local = combined_weights[mask]
            
            X_poly = np.ones((len(X_local), self.degree + 1))
            for d in range(1, self.degree + 1):
                X_poly[:, d] = (X_local - x0)**d
            
            W = np.diag(weights_local)
            XWX = X_poly.T @ W @ X_poly
            XWy = X_poly.T @ W @ y_local
            
            try:
                beta = np.linalg.solve(XWX, XWy)
                pred = beta[0]
            except np.linalg.LinAlgError:
                break
        
        return pred

def demonstrate_local_regression():
    """Demonstrate local regression fitting"""
    # Generate synthetic data
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    y_true = 2 + 3*np.sin(X) + 0.5*X
    y = y_true + np.random.normal(0, 0.5, 100)
    
    # Test different parameters
    models = {}
    
    # Different bandwidths
    for nn_frac in [0.1, 0.3, 0.5]:
        model = LocalRegression(degree=1, nn_frac=nn_frac, robust=False)
        model.fit(X, y)
        models[f'NN={nn_frac}'] = model
    
    # Different degrees
    for degree in [0, 1, 2]:
        model = LocalRegression(degree=degree, nn_frac=0.3, robust=False)
        model.fit(X, y)
        models[f'Degree={degree}'] = model
    
    # Robust vs non-robust
    model_robust = LocalRegression(degree=1, nn_frac=0.3, robust=True)
    model_robust.fit(X, y)
    models['Robust'] = model_robust
    
    # Evaluate models
    X_plot = np.linspace(0, 10, 200)
    
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Effect of bandwidth
    plt.subplot(3, 2, 1)
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.plot(X, y_true, 'k--', label='True Function', linewidth=2)
    
    for name, model in models.items():
        if 'NN=' in name:
            y_plot = model.predict(X_plot)
            plt.plot(X_plot, y_plot, label=name, linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Effect of Bandwidth (Nearest Neighbor Fraction)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Effect of polynomial degree
    plt.subplot(3, 2, 2)
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.plot(X, y_true, 'k--', label='True Function', linewidth=2)
    
    for name, model in models.items():
        if 'Degree=' in name:
            y_plot = model.predict(X_plot)
            plt.plot(X_plot, y_plot, label=name, linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Effect of Polynomial Degree')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Robust vs non-robust
    plt.subplot(3, 2, 3)
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.plot(X, y_true, 'k--', label='True Function', linewidth=2)
    
    # Non-robust
    model_std = models['Degree=1']
    y_plot_std = model_std.predict(X_plot)
    plt.plot(X_plot, y_plot_std, label='Standard', linewidth=2)
    
    # Robust
    y_plot_rob = model_robust.predict(X_plot)
    plt.plot(X_plot, y_plot_rob, label='Robust (LOWESS)', linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Robust vs Non-Robust Fitting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Cross-validation for bandwidth selection
    plt.subplot(3, 2, 4)
    nn_fractions = np.linspace(0.05, 0.8, 20)
    cv_scores = []
    
    for nn_frac in nn_fractions:
        model = LocalRegression(degree=1, nn_frac=nn_frac, robust=False)
        model.fit(X, y)
        
        # Leave-one-out cross-validation
        cv_preds = []
        for i in range(len(X)):
            X_cv = np.delete(X, i)
            y_cv = np.delete(y, i)
            model_cv = LocalRegression(degree=1, nn_frac=nn_frac, robust=False)
            model_cv.fit(X_cv, y_cv)
            pred = model_cv.predict([X[i]])[0]
            cv_preds.append(pred)
        
        cv_score = mean_squared_error(y, cv_preds)
        cv_scores.append(cv_score)
    
    plt.plot(nn_fractions, cv_scores, 'bo-')
    plt.xlabel('Nearest Neighbor Fraction')
    plt.ylabel('Cross-Validation MSE')
    plt.title('Bandwidth Selection via Cross-Validation')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Kernel functions
    plt.subplot(3, 2, 5)
    u = np.linspace(-2, 2, 100)
    
    kernels = ['tricube', 'gaussian', 'epanechnikov']
    for kernel in kernels:
        model = LocalRegression(kernel=kernel)
        weights = model.kernel_function(u)
        plt.plot(u, weights, label=kernel.capitalize(), linewidth=2)
    
    plt.xlabel('u')
    plt.ylabel('K(u)')
    plt.title('Kernel Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Local weights at a point
    plt.subplot(3, 2, 6)
    x0 = 5.0
    model = models['NN=0.3']
    bandwidth = model.compute_bandwidth(X, x0)
    
    distances = np.abs(X - x0)
    u = distances / bandwidth
    weights = model.kernel_function(u)
    
    plt.scatter(X, weights, alpha=0.6)
    plt.axvline(x=x0, color='r', linestyle='--', label=f'x₀ = {x0}')
    plt.xlabel('X')
    plt.ylabel('Weight')
    plt.title(f'Local Weights at x₀ = {x0}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return models

def analyze_outliers():
    """Analyze local regression with outliers"""
    # Generate data with outliers
    np.random.seed(42)
    X = np.linspace(0, 10, 80)
    y_true = 2 + 3*np.sin(X) + 0.5*X
    y = y_true + np.random.normal(0, 0.3, 80)
    
    # Add outliers
    outlier_indices = [20, 40, 60]
    y[outlier_indices] += 3 * np.random.normal(0, 1, len(outlier_indices))
    
    # Fit models
    model_std = LocalRegression(degree=1, nn_frac=0.3, robust=False)
    model_robust = LocalRegression(degree=1, nn_frac=0.3, robust=True)
    
    model_std.fit(X, y)
    model_robust.fit(X, y)
    
    # Predictions
    X_plot = np.linspace(0, 10, 200)
    y_plot_std = model_std.predict(X_plot)
    y_plot_rob = model_robust.predict(X_plot)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.scatter(X[outlier_indices], y[outlier_indices], 
               color='red', s=100, label='Outliers', zorder=5)
    plt.plot(X, y_true, 'k--', label='True Function', linewidth=2)
    plt.plot(X_plot, y_plot_std, label='Standard Local Regression', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Standard Local Regression with Outliers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.scatter(X[outlier_indices], y[outlier_indices], 
               color='red', s=100, label='Outliers', zorder=5)
    plt.plot(X, y_true, 'k--', label='True Function', linewidth=2)
    plt.plot(X_plot, y_plot_rob, label='Robust Local Regression', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Robust Local Regression (LOWESS) with Outliers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(X_plot, y_plot_std, label='Standard', linewidth=2)
    plt.plot(X_plot, y_plot_rob, label='Robust', linewidth=2)
    plt.plot(X, y_true, 'k--', label='True Function', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Comparison of Fits')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Show residuals
    y_pred_std = model_std.predict(X)
    y_pred_rob = model_robust.predict(X)
    
    residuals_std = y - y_pred_std
    residuals_rob = y - y_pred_rob
    
    plt.scatter(y_pred_std, residuals_std, alpha=0.6, label='Standard')
    plt.scatter(y_pred_rob, residuals_rob, alpha=0.6, label='Robust')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return model_std, model_robust

if __name__ == "__main__":
    print("Demonstrating Local Regression...")
    models = demonstrate_local_regression()
    
    print("\nAnalyzing Outliers...")
    model_std, model_robust = analyze_outliers()
