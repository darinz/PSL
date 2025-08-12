"""
Advanced Regression Spline Utilities
===================================

This module provides advanced utilities for regression splines including
model selection with information criteria, regularization methods, and comprehensive diagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

def select_optimal_df_information_criteria(X, y, max_df=20, spline_type='cubic'):
    """
    Select optimal degrees of freedom using information criteria
    
    Parameters:
    X: predictor variable
    y: response variable
    max_df: maximum degrees of freedom to consider
    spline_type: 'cubic' or 'natural'
    
    Returns:
    optimal_df_aic: optimal DF based on AIC
    optimal_df_bic: optimal DF based on BIC
    aic_scores: AIC scores for all DF values
    bic_scores: BIC scores for all DF values
    """
    df_values = range(3, max_df + 1)
    aic_scores = []
    bic_scores = []
    
    for df in df_values:
        model = RegressionSpline(df=df, spline_type=spline_type)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        rss = np.sum((y - y_pred)**2)
        n = len(y)
        
        # AIC
        aic = n * np.log(rss/n) + 2 * df
        aic_scores.append(aic)
        
        # BIC
        bic = n * np.log(rss/n) + df * np.log(n)
        bic_scores.append(bic)
    
    optimal_df_aic = df_values[np.argmin(aic_scores)]
    optimal_df_bic = df_values[np.argmin(bic_scores)]
    
    return optimal_df_aic, optimal_df_bic, aic_scores, bic_scores

def compare_regularization_methods(X, y, df=10):
    """
    Compare different regularization methods
    
    Parameters:
    X: predictor variable
    y: response variable
    df: degrees of freedom
    
    Returns:
    results: dictionary with regularization results
    """
    lambda_values = [0.001, 0.01, 0.1, 1.0, 10.0]
    
    results = {}
    
    for lambda_val in lambda_values:
        # Ridge regression
        model_ridge = RegressionSpline(df=df, regularization='ridge', lambda_val=lambda_val)
        model_ridge.fit(X, y)
        
        # Lasso regression
        model_lasso = RegressionSpline(df=df, regularization='lasso', lambda_val=lambda_val)
        model_lasso.fit(X, y)
        
        # Evaluate
        y_pred_ridge = model_ridge.predict(X)
        y_pred_lasso = model_lasso.predict(X)
        
        mse_ridge = mean_squared_error(y, y_pred_ridge)
        mse_lasso = mean_squared_error(y, y_pred_lasso)
        
        results[lambda_val] = {
            'ridge_mse': mse_ridge,
            'lasso_mse': mse_lasso,
            'ridge_coef': model_ridge.coefficients,
            'lasso_coef': model_lasso.coefficients
        }
    
    return results

def compute_confidence_intervals(model, X, y, X_new, confidence=0.95):
    """
    Compute confidence intervals for regression spline predictions
    
    Parameters:
    model: fitted regression spline model
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
    y_fit = model.predict(X)
    residuals = y - y_fit
    sigma_hat = np.std(residuals)
    
    # Compute leverage
    basis_matrix = model.create_basis_matrix(X)
    basis_new = model.create_basis_matrix(X_new)
    
    H = basis_matrix @ np.linalg.inv(basis_matrix.T @ basis_matrix) @ basis_matrix.T
    leverage = np.diag(H)
    
    # Standard error of prediction
    se_pred = sigma_hat * np.sqrt(1 + np.sum(basis_new**2, axis=1))
    
    # Confidence interval
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha/2, len(y) - model.df)
    
    ci_lower = y_pred - t_critical * se_pred
    ci_upper = y_pred + t_critical * se_pred
    
    return y_pred, ci_lower, ci_upper

def comprehensive_spline_diagnostics(model, X, y):
    """
    Comprehensive diagnostics for regression splines
    
    Parameters:
    model: fitted regression spline model
    X: predictor variable
    y: response variable
    
    Returns:
    residuals: model residuals
    """
    y_pred = model.predict(X)
    residuals = y - y_pred
    
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
    
    # Histogram of residuals
    axes[1, 0].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Histogram of Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scale-Location plot
    axes[1, 1].scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.6)
    axes[1, 1].set_xlabel('Fitted Values')
    axes[1, 1].set_ylabel('√|Residuals|')
    axes[1, 1].set_title('Scale-Location Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Cook's Distance
    basis_matrix = model.create_basis_matrix(X)
    H = basis_matrix @ np.linalg.inv(basis_matrix.T @ basis_matrix) @ basis_matrix.T
    leverage = np.diag(H)
    cooks_d = residuals**2 * leverage / (model.df * np.var(residuals) * (1 - leverage)**2)
    
    axes[1, 2].scatter(range(len(cooks_d)), cooks_d, alpha=0.6)
    axes[1, 2].axhline(y=4/len(X), color='r', linestyle='--', label='4/n threshold')
    axes[1, 2].set_xlabel('Observation Index')
    axes[1, 2].set_ylabel("Cook's Distance")
    axes[1, 2].set_title("Cook's Distance")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return residuals, cooks_d

def demonstrate_advanced_features():
    """Demonstrate advanced regression spline features"""
    # Generate synthetic data
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    y_true = 2 + 3*np.sin(X) + 0.5*X
    y = y_true + np.random.normal(0, 0.3, 100)
    
    # Import RegressionSpline class
    from .regression_spline_implementation import RegressionSpline
    
    # Demonstrate information criteria
    optimal_df_aic, optimal_df_bic, aic_scores, bic_scores = select_optimal_df_information_criteria(X, y)
    
    # Demonstrate regularization comparison
    reg_results = compare_regularization_methods(X, y, df=10)
    
    # Fit best model
    best_model = RegressionSpline(df=optimal_df_aic, spline_type='cubic')
    best_model.fit(X, y)
    
    # Demonstrate confidence intervals
    X_new = np.linspace(0, 10, 200)
    y_pred, ci_lower, ci_upper = compute_confidence_intervals(best_model, X, y, X_new)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Information criteria
    plt.subplot(2, 3, 1)
    df_values = range(3, 21)
    plt.plot(df_values, aic_scores, 'b-', label='AIC')
    plt.plot(df_values, bic_scores, 'r-', label='BIC')
    plt.axvline(x=optimal_df_aic, color='b', linestyle='--', label=f'AIC: DF={optimal_df_aic}')
    plt.axvline(x=optimal_df_bic, color='r', linestyle='--', label=f'BIC: DF={optimal_df_bic}')
    plt.xlabel('Degrees of Freedom')
    plt.ylabel('Information Criterion')
    plt.title('Model Selection: Information Criteria')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Regularization comparison
    plt.subplot(2, 3, 2)
    lambda_values = list(reg_results.keys())
    ridge_mses = [reg_results[lam]['ridge_mse'] for lam in lambda_values]
    lasso_mses = [reg_results[lam]['lasso_mse'] for lam in lambda_values]
    
    plt.semilogx(lambda_values, ridge_mses, 'b-o', label='Ridge')
    plt.semilogx(lambda_values, lasso_mses, 'r-s', label='Lasso')
    plt.xlabel('λ')
    plt.ylabel('MSE')
    plt.title('Regularization Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Confidence intervals
    plt.subplot(2, 3, 3)
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.plot(X_new, y_pred, 'b-', label='Fitted', linewidth=2)
    plt.fill_between(X_new, ci_lower, ci_upper, alpha=0.3, label='95% CI')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Regression Spline with Confidence Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Coefficient comparison
    plt.subplot(2, 3, 4)
    lambda_val = 1.0
    ridge_coef = reg_results[lambda_val]['ridge_coef']
    lasso_coef = reg_results[lambda_val]['lasso_coef']
    
    plt.plot(range(len(ridge_coef)), ridge_coef, 'b-o', label='Ridge')
    plt.plot(range(len(lasso_coef)), lasso_coef, 'r-s', label='Lasso')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Coefficient Value')
    plt.title(f'Coefficient Comparison (λ={lambda_val})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Cross-validation vs Information criteria
    plt.subplot(2, 3, 5)
    cv_scores = []
    for df in df_values:
        model = RegressionSpline(df=df, spline_type='cubic')
        cv_score = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        cv_scores.append(-cv_score.mean())
    
    plt.plot(df_values, cv_scores, 'g-', label='CV MSE')
    plt.axvline(x=optimal_df_aic, color='b', linestyle='--', label=f'AIC: DF={optimal_df_aic}')
    plt.axvline(x=optimal_df_bic, color='r', linestyle='--', label=f'BIC: DF={optimal_df_bic}')
    plt.xlabel('Degrees of Freedom')
    plt.ylabel('Cross-Validation MSE')
    plt.title('CV vs Information Criteria')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Model comparison summary
    plt.subplot(2, 3, 6)
    methods = ['OLS', 'Ridge', 'Lasso']
    mses = [
        mean_squared_error(y, RegressionSpline(df=optimal_df_aic).fit(X, y).predict(X)),
        reg_results[1.0]['ridge_mse'],
        reg_results[1.0]['lasso_mse']
    ]
    
    plt.bar(methods, mses, color=['blue', 'green', 'red'])
    plt.ylabel('MSE')
    plt.title('Model Comparison Summary')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'optimal_df_aic': optimal_df_aic,
        'optimal_df_bic': optimal_df_bic,
        'reg_results': reg_results,
        'best_model': best_model
    }

if __name__ == "__main__":
    results = demonstrate_advanced_features()
