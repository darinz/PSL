"""
Advanced Local Regression Utilities
==================================

This module provides advanced utilities for local regression including
confidence intervals, adaptive bandwidth methods, and comprehensive diagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from scipy import stats

def compute_confidence_intervals(model, X, y, X_new, confidence=0.95):
    """
    Compute confidence intervals for local regression predictions using bootstrap
    
    Parameters:
    model: fitted local regression model
    X: predictor variable
    y: response variable
    X_new: new predictor values for prediction
    confidence: confidence level (default: 0.95)
    
    Returns:
    predictions: predicted values
    ci_lower: lower confidence interval bounds
    ci_upper: upper confidence interval bounds
    """
    predictions = model.predict(X_new)
    
    # Bootstrap confidence intervals
    n_bootstrap = 1000
    bootstrap_preds = np.zeros((n_bootstrap, len(X_new)))
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(len(X), len(X), replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        
        # Fit model on bootstrap sample
        model_boot = type(model)(degree=model.degree, 
                               nn_frac=model.nn_frac, 
                               kernel=model.kernel,
                               robust=model.robust)
        model_boot.fit(X_boot, y_boot)
        
        # Predict
        bootstrap_preds[i, :] = model_boot.predict(X_new)
    
    # Compute confidence intervals
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_preds, alpha/2 * 100, axis=0)
    ci_upper = np.percentile(bootstrap_preds, (1 - alpha/2) * 100, axis=0)
    
    return predictions, ci_lower, ci_upper

def adaptive_bandwidth(X, y, x0, pilot_bandwidth=0.3, alpha=0.5):
    """
    Compute adaptive bandwidth using pilot estimate
    
    Parameters:
    X: predictor variable
    y: response variable
    x0: prediction point
    pilot_bandwidth: bandwidth for pilot estimate
    alpha: adaptation parameter
    
    Returns:
    adaptive_bandwidth: computed adaptive bandwidth
    """
    # Import LocalRegression class
    from .local_regression_implementation import LocalRegression
    
    # Pilot estimate
    pilot_model = LocalRegression(degree=1, nn_frac=pilot_bandwidth)
    pilot_model.fit(X, y)
    pilot_pred = pilot_model.predict(X)
    
    # Compute residuals
    residuals = np.abs(y - pilot_pred)
    
    # Local standard deviation
    distances = np.abs(X - x0)
    weights = pilot_model.kernel_function(distances / pilot_bandwidth)
    local_std = np.sqrt(np.average(residuals**2, weights=weights))
    
    # Global standard deviation
    global_std = np.std(residuals)
    
    # Adaptive bandwidth
    adaptive_factor = (local_std / global_std)**alpha
    base_bandwidth = pilot_model.compute_bandwidth(X, x0)
    
    return base_bandwidth * adaptive_factor

def local_regression_diagnostics(model, X, y):
    """
    Comprehensive diagnostics for local regression
    
    Parameters:
    model: fitted local regression model
    X: predictor variable
    y: response variable
    
    Returns:
    residuals: model residuals
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

def compare_bandwidth_methods(X, y):
    """
    Compare different bandwidth selection methods
    
    Parameters:
    X: predictor variable
    y: response variable
    
    Returns:
    results: dictionary with comparison results
    """
    from .local_regression_implementation import LocalRegression
    
    # Test different bandwidth methods
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
    
    # Find optimal bandwidth
    optimal_nn_frac = nn_fractions[np.argmin(cv_scores)]
    
    # Test adaptive bandwidth
    X_plot = np.linspace(X.min(), X.max(), 100)
    adaptive_preds = []
    
    for x0 in X_plot:
        h_adaptive = adaptive_bandwidth(X, y, x0)
        model_adaptive = LocalRegression(degree=1, bandwidth=h_adaptive)
        model_adaptive.fit(X, y)
        pred = model_adaptive.predict([x0])[0]
        adaptive_preds.append(pred)
    
    return {
        'nn_fractions': nn_fractions,
        'cv_scores': cv_scores,
        'optimal_nn_frac': optimal_nn_frac,
        'X_plot': X_plot,
        'adaptive_preds': adaptive_preds
    }

def demonstrate_advanced_features():
    """Demonstrate advanced local regression features"""
    # Generate synthetic data
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    y_true = 2 + 3*np.sin(X) + 0.5*X
    y = y_true + np.random.normal(0, 0.5, 100)
    
    # Import LocalRegression class
    from .local_regression_implementation import LocalRegression
    
    # Fit standard model
    model = LocalRegression(degree=1, nn_frac=0.3, robust=False)
    model.fit(X, y)
    
    # Demonstrate confidence intervals
    X_new = np.linspace(0, 10, 200)
    y_pred, ci_lower, ci_upper = compute_confidence_intervals(model, X, y, X_new)
    
    # Demonstrate adaptive bandwidth
    bandwidth_results = compare_bandwidth_methods(X, y)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Confidence intervals
    plt.subplot(2, 3, 1)
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.plot(X_new, y_pred, 'b-', label='Fitted', linewidth=2)
    plt.fill_between(X_new, ci_lower, ci_upper, alpha=0.3, label='95% CI')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Local Regression with Confidence Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Bandwidth selection
    plt.subplot(2, 3, 2)
    plt.plot(bandwidth_results['nn_fractions'], bandwidth_results['cv_scores'], 'bo-')
    plt.axvline(x=bandwidth_results['optimal_nn_frac'], color='r', linestyle='--', 
                label=f'Optimal: {bandwidth_results["optimal_nn_frac"]:.2f}')
    plt.xlabel('Nearest Neighbor Fraction')
    plt.ylabel('Cross-Validation MSE')
    plt.title('Bandwidth Selection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Adaptive vs fixed bandwidth
    plt.subplot(2, 3, 3)
    plt.scatter(X, y, alpha=0.6, label='Data')
    
    # Fixed bandwidth
    model_fixed = LocalRegression(degree=1, nn_frac=0.3)
    model_fixed.fit(X, y)
    y_fixed = model_fixed.predict(bandwidth_results['X_plot'])
    plt.plot(bandwidth_results['X_plot'], y_fixed, 'b-', label='Fixed Bandwidth', linewidth=2)
    
    # Adaptive bandwidth
    plt.plot(bandwidth_results['X_plot'], bandwidth_results['adaptive_preds'], 
             'r-', label='Adaptive Bandwidth', linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Fixed vs Adaptive Bandwidth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Kernel comparison
    plt.subplot(2, 3, 4)
    kernels = ['tricube', 'gaussian', 'epanechnikov']
    u = np.linspace(-2, 2, 100)
    
    for kernel in kernels:
        model_kernel = LocalRegression(kernel=kernel)
        weights = model_kernel.kernel_function(u)
        plt.plot(u, weights, label=kernel.capitalize(), linewidth=2)
    
    plt.xlabel('u')
    plt.ylabel('K(u)')
    plt.title('Kernel Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Robust vs non-robust comparison
    plt.subplot(2, 3, 5)
    plt.scatter(X, y, alpha=0.6, label='Data')
    
    # Non-robust
    model_std = LocalRegression(degree=1, nn_frac=0.3, robust=False)
    model_std.fit(X, y)
    y_std = model_std.predict(X_new)
    plt.plot(X_new, y_std, 'b-', label='Standard', linewidth=2)
    
    # Robust
    model_rob = LocalRegression(degree=1, nn_frac=0.3, robust=True)
    model_rob.fit(X, y)
    y_rob = model_rob.predict(X_new)
    plt.plot(X_new, y_rob, 'r-', label='Robust (LOWESS)', linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Standard vs Robust Fitting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Model diagnostics
    plt.subplot(2, 3, 6)
    residuals = local_regression_diagnostics(model, X, y)
    
    # Show residual statistics
    plt.text(0.1, 0.9, f'Mean Residual: {np.mean(residuals):.3f}', 
             transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.1, 0.8, f'Std Residual: {np.std(residuals):.3f}', 
             transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.1, 0.7, f'MSE: {mean_squared_error(y, model.predict(X)):.3f}', 
             transform=plt.gca().transAxes, fontsize=10)
    plt.axis('off')
    plt.title('Model Summary')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'model': model,
        'confidence_intervals': (y_pred, ci_lower, ci_upper),
        'bandwidth_results': bandwidth_results
    }

if __name__ == "__main__":
    results = demonstrate_advanced_features()
